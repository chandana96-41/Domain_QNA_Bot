"""
Domain RAG chatbot: upload documents, index with Ollama embeddings, chat with Ollama LLM.
Run: streamlit run app.py
"""
import os
from pathlib import Path

import streamlit as st

from rag.config import UPLOADS_DIR, ensure_dirs
from rag.ingest import clear_index, ingest_files
from rag.chat import answer_question, get_vectorstore
from rag.ollama_health import require_models

st.set_page_config(page_title="Domain RAG (Ollama)", page_icon="📚", layout="wide")
ensure_dirs()

st.title("Domain Q&A with RAG")
st.caption("Upload manuals, PDFs, or text. Index them, then ask questions. Uses Ollama on your machine.")

with st.sidebar:
    st.header("Settings")
    ollama_base = st.text_input("Ollama URL", value="http://127.0.0.1:11434")
    embed_model = st.text_input("Embedding model", value="nomic-embed-text", help="Run: ollama pull nomic-embed-text")
    chat_model = st.text_input("Chat model", value="llama3.2", help="Run: ollama pull llama3.2")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=12, value=4)

    st.divider()
    st.subheader("Documents")
    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )
    replace_index = st.checkbox("Replace existing index (wipe old docs)", value=False)

    if st.button("Index uploaded documents", type="primary"):
        if not uploaded:
            st.warning("Choose at least one file.")
        else:
            base_url = ollama_base.strip() or "http://127.0.0.1:11434"
            os.environ["OLLAMA_BASE_URL"] = base_url
            try:
                require_models(base_url, [embed_model])
            except Exception as e:
                st.error(str(e))
                st.stop()
            paths: list[Path] = []
            for f in uploaded:
                dest = UPLOADS_DIR / f.name
                dest.write_bytes(f.getvalue())
                paths.append(dest)
            with st.spinner("Chunking and embedding (first run can be slow)..."):
                try:
                    n = ingest_files(paths, embed_model=embed_model or None, replace=replace_index)
                    st.success(f"Indexed {n} chunks from {len(paths)} file(s).")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    if st.button("Clear index"):
        clear_index()
        st.success("Index cleared.")
        st.rerun()

    vs = get_vectorstore(embed_model)
    if vs is not None:
        st.info("Index ready — you can chat.")
    else:
        st.warning("No index yet — upload and index documents first.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.text(s)

if prompt := st.chat_input("Ask about your documents..."):
    base_url = ollama_base.strip() or "http://127.0.0.1:11434"
    os.environ["OLLAMA_BASE_URL"] = base_url
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                require_models(base_url, [embed_model, chat_model])
                answer, sources = answer_question(
                    prompt,
                    embed_model=embed_model or None,
                    chat_model=chat_model or None,
                    k=top_k,
                )
            except Exception as e:
                answer = f"Error: {e}"
                sources = []
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.text(s)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()
