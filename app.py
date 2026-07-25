import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np
import faiss

# ========== PAGE SETUP ==========
st.set_page_config(page_title="My RAG App", page_icon="📚", layout="centered")

st.title("📚 My Free RAG Application")
st.markdown("""
**Upload PDF documents → Ask questions → Get AI answers based ONLY on your documents**

*Built with 100% free tools: Streamlit + Gemini API + FAISS*
""")

# ========== GEMINI SETUP ==========
# For local: use environment variable or .env
# For Streamlit Cloud: uses st.secrets
import os

try:
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ Gemini API Key not found!")
        st.info("""
        **To fix this:**
        - **Local**: Create a `.env` file with `GEMINI_API_KEY=your_key`
        - **Streamlit Cloud**: Go to app settings → Secrets → Add `GEMINI_API_KEY`
        """)
        st.stop()
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring Gemini: {e}")
    st.stop()

# ========== HELPER FUNCTIONS ==========

def get_embedding(text: str) -> np.ndarray:
    """Convert text to vector using Gemini Embedding (FREE tier)"""
    try:
        # Gemini has a content limit, so we truncate safely
        safe_text = text[:8000] if len(text) > 8000 else text
        result = genai.embed_content(
            model="models/embedding-001",
            content=safe_text,
            task_type="retrieval_document"
        )
        return np.array(result['embedding'], dtype=np.float32)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def get_answer(question: str, context: str) -> str:
    """Ask Gemini Flash to answer based on retrieved context"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""You are a helpful study assistant. Answer the question using ONLY the information provided in the context below.
If the answer is not found in the context, say: "I don't have enough information in the uploaded documents to answer this."

=== CONTEXT FROM DOCUMENTS ===
{context}

=== USER QUESTION ===
{question}

=== YOUR ANSWER ===
Provide a clear, accurate, and concise answer."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating answer: {e}"

def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list:
    """Split long text into overlapping chunks for better retrieval"""
    if not text or len(text) < chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at a natural boundary (period, newline, or space)
        if end < text_len:
            for sep in ['. ', '\n', ' ']:
                pos = text.rfind(sep, start, end)
                if pos != -1:
                    end = pos + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk and len(chunk) > 50:  # Ignore tiny fragments
            chunks.append(chunk)

        start = end - overlap  # Overlap for continuity

    return chunks

def extract_pdf_text(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        reader = PdfReader(pdf_file)
        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(f"[Page {i+1}]\n{page_text}")
        return "\n\n".join(text_parts)
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {e}")
        return ""

# ========== MAIN APP ==========

# File uploader
uploaded_files = st.file_uploader(
    "📄 Upload your PDF files (you can select multiple)",
    type=['pdf'],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👆 **Get started:** Upload one or more PDF files above")
    st.markdown("""
    ### 💡 What is RAG?
    **RAG** = **R**etrieval **A**ugmented **G**eneration

    1. 📄 You upload documents
    2. 🔍 The app finds the most relevant parts
    3. 🤖 AI reads those parts and answers your question
    4. ✅ You get accurate answers based on YOUR documents, not random internet knowledge
    """)
    st.stop()

# Process documents button
if st.button("🚀 Process Documents", type="primary"):
    with st.spinner("🔍 Reading and analyzing your documents..."):

        # Step 1: Extract text from all PDFs
        all_chunks = []
        file_stats = []

        progress = st.progress(0)
        for idx, pdf_file in enumerate(uploaded_files):
            text = extract_pdf_text(pdf_file)
            if text:
                chunks = split_text(text)
                all_chunks.extend(chunks)
                file_stats.append({
                    'name': pdf_file.name,
                    'pages': len(PdfReader(pdf_file).pages),
                    'chunks': len(chunks)
                })
            progress.progress((idx + 1) / len(uploaded_files))

        if not all_chunks:
            st.error("❌ Could not extract text from the uploaded PDFs. Try different files.")
            st.stop()

        # Step 2: Create embeddings (vectors) for each chunk
        st.write(f"🧠 Creating {len(all_chunks)} embeddings... (this may take a moment)")
        embed_progress = st.progress(0)
        embeddings = []

        for i, chunk in enumerate(all_chunks):
            emb = get_embedding(chunk)
            if emb is not None:
                embeddings.append(emb)
            embed_progress.progress((i + 1) / len(all_chunks))

        if not embeddings:
            st.error("❌ Failed to create embeddings. Check your API key.")
            st.stop()

        # Step 3: Build FAISS vector index (in-memory, fast & free)
        embeddings = np.array(embeddings)
        dimension = embeddings.shape[1]  # Usually 768 for Gemini embeddings

        index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
        index.add(embeddings)

        # Store in session state so it persists while app is running
        st.session_state['index'] = index
        st.session_state['chunks'] = all_chunks
        st.session_state['file_stats'] = file_stats
        st.session_state['ready'] = True

        st.success(f"✅ Ready! Processed {len(uploaded_files)} file(s) into {len(all_chunks)} searchable chunks.")

# Show file stats in sidebar
with st.sidebar:
    st.header("📊 Document Stats")
    if 'file_stats' in st.session_state:
        for stat in st.session_state['file_stats']:
            st.markdown(f"**{stat['name']}**")
            st.markdown(f"- Pages: {stat['pages']} | Chunks: {stat['chunks']}")
    else:
        st.info("Upload and process documents to see stats")

    st.divider()
    st.markdown("""
    ### 🛠️ Tech Stack
    - **UI**: Streamlit (Free)
    - **AI Model**: Gemini 2.0 Flash (Free tier)
    - **Embeddings**: Gemini Embedding (Free tier)
    - **Vector DB**: FAISS (Open source)
    - **PDF Reader**: PyPDF (Open source)
    """)

# Question & Answer section
if st.session_state.get('ready', False):
    st.divider()
    st.subheader("❓ Ask Your Documents")

    question = st.text_input(
        "Type your question here:",
        placeholder="e.g., What are the main findings? Who is the author? What does chapter 3 say about...?"
    )

    if question:
        with st.spinner("🔎 Searching documents and generating answer..."):
            # 1. Embed the question
            q_emb = get_embedding(question)
            if q_emb is None:
                st.stop()

            q_emb = np.array([q_emb])

            # 2. Search FAISS for top 3 most similar chunks
            k = min(3, len(st.session_state['chunks']))
            distances, indices = st.session_state['index'].search(q_emb, k)

            # 3. Retrieve the actual text chunks
            relevant_chunks = [st.session_state['chunks'][i] for i in indices[0]]
            context = "\n\n---\n\n".join(relevant_chunks)

            # 4. Ask Gemini to answer using only the retrieved context
            answer = get_answer(question, context)

            # 5. Display results
            st.markdown("### 💡 Answer")
            st.info(answer)

            # Show source chunks (expandable)
            with st.expander("📄 View source text chunks used to generate this answer"):
                for i, (chunk, dist) in enumerate(zip(relevant_chunks, distances[0])):
                    st.markdown(f"**Relevant Chunk {i+1}** *(relevance score: {dist:.2f})*")
                    st.text_area(f"chunk_{i}", chunk[:800] + ("..." if len(chunk) > 800 else ""), 
                                height=120, label_visibility="collapsed")
                    st.divider()
