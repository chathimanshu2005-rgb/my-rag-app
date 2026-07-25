import streamlit as st
from google import genai
from pypdf import PdfReader
import numpy as np

# ========== PAGE SETUP ==========
st.set_page_config(page_title="My RAG App", page_icon="📚", layout="centered")

st.title("📚 My Free RAG Application")
st.markdown("**Upload PDF documents → Ask questions → Get AI answers based ONLY on your documents**")

# ========== GEMINI SETUP (NO st.stop() ANYWHERE) ==========
import os

api_key = None
client = None
api_status = "❌ Not connected"

# Try to get API key - very defensive
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
        api_status = "✅ Connected"
    except Exception as e:
        api_status = f"❌ Connection failed: {e}"
else:
    api_status = "❌ API Key missing"

# Show API status in sidebar
with st.sidebar:
    st.header("🔌 API Status")
    st.write(api_status)

    if not api_key:
        st.markdown("""
        **Add your API key:**
        1. Go to [share.streamlit.io](https://share.streamlit.io)
        2. Click ⋮ → **Settings** → **Secrets**
        3. Add: `GEMINI_API_KEY = "your-key"`
        4. Click **Save** and **Reboot**
        """)

    st.divider()
    st.markdown("""
    ### 🛠️ Tech Stack
    - **UI**: Streamlit (Free)
    - **AI**: Gemini 2.0 Flash (Free tier)
    - **Embeddings**: Gemini (Free tier)
    - **Search**: Pure Python
    - **PDF**: PyPDF
    """)

# If no API key, show setup page and RETURN (no st.stop!)
if not client:
    st.warning("⚠️ Please add your Gemini API key in the sidebar instructions above.")
    st.info("👇 Once added, this page will show the full RAG app.")
    st.markdown("""
    ### 💡 What is RAG?
    **RAG** = **R**etrieval **A**ugmented **G**eneration

    1. 📄 You upload documents  
    2. 🔍 The app finds the most relevant parts  
    3. 🤖 AI reads those parts and answers your question  
    4. ✅ You get accurate answers based on YOUR documents
    """)
    # END OF SCRIPT - no st.stop(), just natural end
else:
    # ========== MAIN APP (only runs if API key works) ==========

    uploaded_files = st.file_uploader(
        "📄 Upload your PDF files (you can select multiple)",
        type=['pdf'],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("👆 **Get started:** Upload one or more PDF files above")
    else:
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("🔍 Reading and analyzing your documents..."):

                all_chunks = []
                file_stats = []

                progress = st.progress(0)
                for idx, pdf_file in enumerate(uploaded_files):
                    try:
                        reader = PdfReader(pdf_file)
                        text_parts = []
                        for i, page in enumerate(reader.pages):
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                text_parts.append(f"[Page {i+1}]\n{page_text}")
                        full_text = "\n\n".join(text_parts)
                        page_count = len(reader.pages)

                        if full_text:
                            # Split into chunks
                            chunks = []
                            start = 0
                            text_len = len(full_text)
                            chunk_size = 1000
                            overlap = 150

                            while start < text_len:
                                end = min(start + chunk_size, text_len)
                                if end < text_len:
                                    for sep in ['. ', '\n', ' ']:
                                        pos = full_text.rfind(sep, start, end)
                                        if pos != -1:
                                            end = pos + len(sep)
                                            break
                                chunk = full_text[start:end].strip()
                                if chunk and len(chunk) > 50:
                                    chunks.append(chunk)
                                start = end - overlap

                            all_chunks.extend(chunks)
                            file_stats.append({
                                'name': pdf_file.name,
                                'pages': page_count,
                                'chunks': len(chunks)
                            })
                    except Exception as e:
                        st.error(f"Error reading {pdf_file.name}: {e}")

                    progress.progress((idx + 1) / len(uploaded_files))

                if not all_chunks:
                    st.error("❌ Could not extract text from PDFs. Try text-based PDFs (not scanned images).")
                else:
                    # Create embeddings
                    st.write(f"🧠 Creating {len(all_chunks)} embeddings...")
                    embed_progress = st.progress(0)
                    embeddings = []

                    for i, chunk in enumerate(all_chunks):
                        try:
                            safe_text = chunk[:8000] if len(chunk) > 8000 else chunk
                            result = client.models.embed_content(
                                model="gemini-embedding-exp-03-07",
                                contents=[safe_text]
                            )
                            emb = result.embeddings[0]
                            if hasattr(emb, 'values'):
                                embeddings.append(np.array(emb.values, dtype=np.float32))
                            else:
                                embeddings.append(np.array(emb, dtype=np.float32))
                        except Exception as e:
                            st.error(f"Embedding error on chunk {i}: {e}")

                        embed_progress.progress((i + 1) / len(all_chunks))

                    if embeddings:
                        st.session_state['embeddings'] = embeddings
                        st.session_state['chunks'] = all_chunks
                        st.session_state['file_stats'] = file_stats
                        st.session_state['ready'] = True
                        st.success(f"✅ Ready! Processed {len(uploaded_files)} file(s) into {len(all_chunks)} chunks.")
                    else:
                        st.error("❌ Failed to create embeddings.")

        # Show file stats
        if 'file_stats' in st.session_state:
            st.subheader("📊 Documents")
            for stat in st.session_state['file_stats']:
                st.write(f"**{stat['name']}** — {stat['pages']} pages, {stat['chunks']} chunks")

        # Question section
        if st.session_state.get('ready', False):
            st.divider()
            st.subheader("❓ Ask Your Documents")

            question = st.text_input(
                "Type your question here:",
                placeholder="e.g., What are the main findings?"
            )

            if question:
                with st.spinner("🔎 Searching and generating answer..."):
                    try:
                        # Embed question
                        safe_q = question[:8000] if len(question) > 8000 else question
                        q_result = client.models.embed_content(
                            model="gemini-embedding-exp-03-07",
                            contents=[safe_q]
                        )
                        q_emb = q_result.embeddings[0]
                        if hasattr(q_emb, 'values'):
                            q_emb = np.array(q_emb.values, dtype=np.float32)
                        else:
                            q_emb = np.array(q_emb, dtype=np.float32)

                        # Find top 3 similar chunks
                        similarities = []
                        for emb in st.session_state['embeddings']:
                            sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
                            similarities.append(sim)

                        top_indices = np.argsort(similarities)[-3:][::-1]

                        # Build context
                        relevant_chunks = [st.session_state['chunks'][i] for i in top_indices]
                        context = "\n\n---\n\n".join(relevant_chunks)

                        # Generate answer
                        prompt = f"""You are a helpful study assistant. Answer the question using ONLY the information provided in the context below.
If the answer is not found in the context, say: "I don't have enough information in the uploaded documents to answer this."

=== CONTEXT ===
{context}

=== QUESTION ===
{question}

=== ANSWER ==="""

                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=prompt
                        )

                        st.markdown("### 💡 Answer")
                        st.info(response.text)

                        with st.expander("📄 View source chunks"):
                            for i, idx in enumerate(top_indices):
                                chunk = st.session_state['chunks'][idx]
                                score = similarities[idx]
                                st.markdown(f"**Chunk {i+1}** (score: {score:.3f})")
                                st.text(chunk[:800] + ("..." if len(chunk) > 800 else ""))
                                st.divider()

                    except Exception as e:
                        st.error(f"❌ Error during Q&A: {e}")
