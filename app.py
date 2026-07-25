import streamlit as st
from google import genai
from pypdf import PdfReader
import numpy as np
import time

st.set_page_config(page_title="My RAG App", page_icon="📚", layout="centered")
st.title("📚 My Free RAG Application")

# ========== API KEY ==========
import os

api_key = None
client = None

try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"API Error: {e}")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("🔌 Status")
    if client:
        st.success("✅ Gemini Connected")
    else:
        st.error("❌ No API Key")
        st.markdown("Add in Streamlit Cloud: Settings → Secrets → `GEMINI_API_KEY = \\\"your-key\\\"`")
    
    st.divider()
    st.markdown("""
    ### ⏱️ Free Tier Limits
    - **15 requests per minute**
    - **1,500 requests per day**
    - If you see "Quota exceeded", wait 30-60 seconds and try again
    """)

# ========== MAIN APP ==========
if not client:
    st.warning("⚠️ Please add your Gemini API key in Streamlit Cloud Secrets.")
else:
    uploaded_files = st.file_uploader(
        "📄 Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.info("👆 Upload PDF files to get started")
    else:
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing..."):
                all_chunks = []
                file_stats = []
                
                for pdf_file in uploaded_files:
                    try:
                        reader = PdfReader(pdf_file)
                        text_parts = []
                        for i, page in enumerate(reader.pages):
                            txt = page.extract_text()
                            if txt:
                                text_parts.append(f"[Page {i+1}]\n{txt}")
                        
                        full_text = "\n\n".join(text_parts)
                        pages = len(reader.pages)
                        
                        chunks = []
                        start = 0
                        while start < len(full_text):
                            end = min(start + 1000, len(full_text))
                            chunk = full_text[start:end].strip()
                            if len(chunk) > 50:
                                chunks.append(chunk)
                            start = end - 150 if end < len(full_text) else end
                        
                        all_chunks.extend(chunks)
                        file_stats.append({"name": pdf_file.name, "pages": pages, "chunks": len(chunks)})
                    except Exception as e:
                        st.error(f"Error with {pdf_file.name}: {e}")
                
                if not all_chunks:
                    st.error("No text extracted. Try text-based PDFs.")
                else:
                    embeddings = []
                    emb_progress = st.progress(0)
                    status = st.empty()
                    
                    for i, chunk in enumerate(all_chunks):
                        try:
                            status.write(f"Embedding chunk {i+1}/{len(all_chunks)}...")
                            chunk_text = chunk[:8000]
                            
                            result = client.models.embed_content(
                                model="gemini-embedding-2",
                                contents=[chunk_text]
                            )
                            
                            emb = result.embeddings[0]
                            val = emb.values if hasattr(emb, 'values') else emb
                            embeddings.append(np.array(val, dtype=np.float32))
                            
                            # ========== FIX: Sleep to avoid rate limit ==========
                            time.sleep(0.5)
                            
                        except Exception as e:
                            error_msg = str(e)
                            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                                st.warning(f"⏳ Rate limit hit on chunk {i}. Waiting 30 seconds...")
                                time.sleep(30)
                                # Retry once
                                try:
                                    result = client.models.embed_content(
                                        model="gemini-embedding-2",
                                        contents=[chunk_text]
                                    )
                                    emb = result.embeddings[0]
                                    val = emb.values if hasattr(emb, 'values') else emb
                                    embeddings.append(np.array(val, dtype=np.float32))
                                except Exception as e2:
                                    st.error(f"Retry failed for chunk {i}: {e2}")
                            else:
                                st.error(f"Embed error chunk {i}: {e}")
                        
                        emb_progress.progress((i + 1) / len(all_chunks))
                    
                    status.empty()
                    
                    if embeddings:
                        st.session_state.embeddings = embeddings
                        st.session_state.chunks = all_chunks
                        st.session_state.file_stats = file_stats
                        st.session_state.ready = True
                        st.success(f"✅ Ready! {len(uploaded_files)} files → {len(all_chunks)} chunks")
                    else:
                        st.error("❌ Failed to create embeddings.")
        
        if "file_stats" in st.session_state:
            for s in st.session_state.file_stats:
                st.write(f"📄 {s['name']} — {s['pages']} pages → {s['chunks']} chunks")
        
        if st.session_state.get("ready"):
            st.divider()
            question = st.text_input("❓ Ask a question about your documents:")
            
            if question:
                with st.spinner("Thinking..."):
                    try:
                        q_text = question[:8000]
                        
                        q_res = client.models.embed_content(
                            model="gemini-embedding-2",
                            contents=[q_text]
                        )
                        
                        q_emb = q_res.embeddings[0]
                        q_vec = np.array(q_emb.values if hasattr(q_emb, 'values') else q_emb, dtype=np.float32)
                        
                        sims = []
                        for emb in st.session_state
