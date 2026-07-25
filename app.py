import streamlit as st

st.set_page_config(page_title="My RAG App", page_icon="📚", layout="centered")
st.title("📚 My Free RAG Application")

# ========== SAFE IMPORTS ==========
import os
import numpy as np

# Try importing google-genai safely
genai_import_error = None
try:
    from google import genai
except Exception as e:
    genai_import_error = str(e)

# Try importing pypdf safely
pypdf_import_error = None
try:
    from pypdf import PdfReader
except Exception as e:
    pypdf_import_error = str(e)

# ========== API KEY ==========
api_key = None
client = None
api_error = None

try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    api_error = f"Secrets error: {e}"

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

if api_key and genai_import_error is None:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        api_error = f"Client creation failed: {e}"

# ========== SIDEBAR STATUS ==========
with st.sidebar:
    st.header("🔌 System Status")

    if genai_import_error:
        st.error(f"❌ google-genai import failed: {genai_import_error}")
    else:
        st.success("✅ google-genai imported")

    if pypdf_import_error:
        st.error(f"❌ pypdf import failed: {pypdf_import_error}")
    else:
        st.success("✅ pypdf imported")

    if api_key:
        st.success(f"✅ API key found ({api_key[:8]}...)")
    else:
        st.error("❌ API key missing")

    if client:
        st.success("✅ Gemini client ready")
    elif api_key and genai_import_error is None:
        st.error(f"❌ Gemini client failed: {api_error}")

    if not api_key:
        st.markdown("""
        **How to add API key:**
        1. Go to [share.streamlit.io](https://share.streamlit.io)
        2. Click ⋮ → Settings → Secrets
        3. Paste exactly:
        ```
        GEMINI_API_KEY = "your-key-here"
        ```
        4. Save & Reboot
        """)

# ========== MAIN APP ==========

# If anything is broken, show diagnostic page
if genai_import_error or pypdf_import_error or not client:
    st.warning("⚠️ App is in diagnostic mode — some features are disabled.")

    if genai_import_error:
        st.error("**google-genai failed to import.** This usually means:")
        st.markdown("""
        - Your `requirements.txt` has the wrong package name
        - It should say: `google-genai>=1.0.0` (NOT `google-generativeai`)
        """)

    if not api_key:
        st.error("**No API key found.** Check Streamlit Cloud Secrets.")

    if api_error and api_key:
        st.error(f"**API connection failed:** {api_error}")

    st.info("Fix the issues above, then reboot the app.")

    # Still show the uploader so they can test PDF reading
    if pypdf_import_error is None:
        st.divider()
        st.subheader("📄 PDF Test (no AI needed)")
        test_pdf = st.file_uploader("Upload a PDF to test extraction:", type=['pdf'])
        if test_pdf:
            try:
                reader = PdfReader(test_pdf)
                text = "\n".join([p.extract_text() or "" for p in reader.pages])
                st.success(f"✅ Extracted {len(text)} characters from {len(reader.pages)} pages")
                st.text(text[:500] + "...")
            except Exception as e:
                st.error(f"PDF read failed: {e}")

else:
    # ========== FULL RAG APP ==========
    uploaded_files = st.file_uploader(
        "📄 Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("👆 Upload PDF files to get started")
        st.markdown("""
        ### How it works:
        1. Upload your PDF documents
        2. Click "Process Documents"
        3. Ask questions — answers come from YOUR documents only
        """)
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
                            safe = chunk[:8000]
                            result = client.models.embed_content(
                                model="gemini-embedding-exp-03-07",
                                contents=[safe]
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

        # Show stats
        if "file_stats" in st.session_state:
            for s in st.session_state.file_stats:
                st.write(f"📄 {s['name']} — {s['pages']} pages → {s['chunks']} chunks")

        # Q&A
        if st.session_state.get("ready"):
            st.divider()
            question = st.text_input("❓ Ask a question about your documents:")

            if question:
                with st.spinner("Thinking..."):
                    try:
                        # Embed question
                        q_res = client.models.embed_content(
                            model="gemini-embedding-exp-03-07",
                            contents=[question[:8000]]
                        )
                        q_emb = q_res.embeddings[0]
                        q_vec = np.array(q_emb.values if hasattr(q_emb, 'values') else q_emb, dtype=np.float32)

                        # Find top 3
                        sims = []
                        for emb in st.session_state.embeddings:
                            sim = np.dot(q_vec, emb) / (np.linalg.norm(q_vec) * np.linalg.norm(emb))
                            sims.append(sim)

                        top_idx = np.argsort(sims)[-3:][::-1]
                        context = "\n\n---\n\n".join([st.session_state.chunks[i] for i in top_idx])

                        # Generate answer
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

