import streamlit as st
from google import genai
from pypdf import PdfReader
import numpy as np

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
                        
                        # Split into chunks
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
                    # Create embeddings
                    embeddings = []
                    emb_progress = st.progress(0)
                    status = st.empty()
                    
                    for i, chunk in enumerate(all_chunks):
                        try:
                            status.write(f"Embedding chunk {i+1}/{len(all_chunks)}...")
                            chunk_text = chunk[:8000]
                            
                            result = client.models.embed_content(
                                model="text-embedding-004",
                                contents=[chunk_text]
                            )
                            
                            emb = result.embeddings[0]
                            val = emb.values if hasattr(emb, 'values') else emb
                            embeddings.append(np.array(val, dtype=np.float32))
                        except Exception as e:
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
                            model="text-embedding-004",
                            contents=[q_text]
                        )
                        q_emb = q_res.embeddings[0]
                        q_vec = np.array(q_emb.values if hasattr(q_emb, 'values') else q_emb, dtype=np.float32)
                        
                        sims = []
                        for emb in st.session_state.embeddings:
                            sim = np.dot(q_vec, emb) / (np.linalg.norm(q_vec) * np.linalg.norm(emb))
                            sims.append(sim)
                        
                        top_idx = np.argsort(sims)[-3:][::-1]
                        context = "\n\n---\n\n".join([st.session_state.chunks[i] for i in top_idx])
                        
                        prompt = f"""Answer using ONLY the context below. If not found, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""
                        
                        ans = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=prompt
                        )
                        
                        st.markdown("### 💡 Answer")
                        st.info(ans.text)
                        
                        with st.expander("📄 Source chunks"):
                            for i, idx in enumerate(top_idx):
                                st.markdown(f"**Chunk {i+1}** (score: {sims[idx]:.3f})")
                                st.text(st.session_state.chunks[idx][:600])
                    except Exception as e:
                        st.error(f"Q&A Error: {e}")
