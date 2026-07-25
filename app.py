import streamlit as st
from pypdf import PdfReader
import numpy as np
from fastembed import TextEmbedding
from groq import Groq

st.set_page_config(page_title="My RAG App", page_icon="📚", layout="centered")
st.title("📚 My Free RAG Application")

# ========== API KEYS ==========
import os

groq_key = None
groq_client = None

try:
    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not groq_key:
    groq_key = os.getenv("GROQ_API_KEY")

if groq_key:
    try:
        groq_client = Groq(api_key=groq_key)
    except Exception as e:
        st.sidebar.error(f"Groq Error: {e}")

# ========== LOCAL EMBEDDING MODEL ==========
@st.cache_resource
def load_embedder():
    with st.spinner("Downloading embedding model (22MB, one-time)..."):
        return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

embedder = None
try:
    embedder = load_embedder()
except Exception as e:
    st.sidebar.error(f"Embedder Error: {e}")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("🔌 Status")
    if groq_client:
        st.success("✅ Groq Connected")
    else:
        st.error("❌ No Groq API Key")
        st.markdown("Get free key at console.groq.com")
    
    if embedder:
        st.success("✅ Local Embedder Ready")
    else:
        st.error("❌ Embedder failed")
    
    st.divider()
    st.markdown("""
    ### ⏱️ Limits
    - **Groq**: 30 requests/minute (free tier)
    - **Embeddings**: Unlimited (runs locally!)
    - No credit card needed
    """)

# ========== MAIN APP ==========
if not groq_client or not embedder:
    st.warning("⚠️ Setup Required")
    st.markdown("""
    ### Step 1: Get Groq API Key (Free, No Credit Card)
    1. Go to https://console.groq.com
    2. Sign up with email or Google
    3. Click "API Keys" → "Create API Key"
    4. Copy your key
    
    ### Step 2: Add to Streamlit Cloud Secrets
    1. Go to share.streamlit.io → your app → Settings → Secrets
    2. Add: `GROQ_API_KEY = "your-key"`
    3. Save & Reboot
    """)
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
                    status = st.empty()
                    status.write("Creating embeddings locally (no API calls, no rate limits)...")
                    
                    try:
                        emb_generator = embedder.embed(all_chunks)
                        embeddings_list = list(emb_generator)
                        embeddings = np.array(embeddings_list, dtype=np.float32)
                        
                        st.session_state.embeddings = embeddings
                        st.session_state.chunks = all_chunks
                        st.session_state.file_stats = file_stats
                        st.session_state.ready = True
                        status.empty()
                        st.success(f"✅ Ready! {len(uploaded_files)} files → {len(all_chunks)} chunks")
                    except Exception as e:
                        st.error(f"Embedding failed: {e}")
        
        if "file_stats" in st.session_state:
            for s in st.session_state.file_stats:
                st.write(f"📄 {s['name']} — {s['pages']} pages → {s['chunks']} chunks")
        
        if st.session_state.get("ready"):
            st.divider()
            question = st.text_input("❓ Ask a question about your documents:")
            
            if question:
                with st.spinner("Thinking..."):
                    try:
                        q_emb_generator = embedder.embed([question])
                        q_vec = np.array(list(q_emb_generator)[0], dtype=np.float32)
                        
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
                        
                        chat_completion = groq_client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="llama-3.3-70b-versatile",
                            temperature=0.3,
                            max_tokens=1024
                        )
                        
                        st.markdown("### 💡 Answer")
                        st.info(chat_completion.choices[0].message.content)
                        
                        with st.expander("📄 Source chunks"):
                            for i, idx in enumerate(top_idx):
                                st.markdown(f"**Chunk {i+1}** (score: {sims[idx]:.3f})")
                                st.text(st.session_state.chunks[idx][:600])
                    
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg:
                            st.error("⏳ Groq rate limit (30/min). Wait a few seconds and try again.")
                        else:
                            st.error(f"Q&A Error: {e}")
