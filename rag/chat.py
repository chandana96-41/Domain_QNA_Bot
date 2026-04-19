"""RAG question answering using Ollama chat + Chroma retrieval."""
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag.config import CHROMA_COLLECTION, CHROMA_DIR, DEFAULT_CHAT_MODEL, ollama_base_url
from rag.ingest import get_embeddings


SYSTEM = """You are a helpful assistant that answers questions using ONLY the context provided below.
If the answer is not contained in the context, say you do not have enough information in the uploaded documents.
Do not invent facts. Cite which part of the context supports your answer when possible.
Context:
{context}"""


def get_vectorstore(embed_model: str | None = None) -> Chroma | None:
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        return None
    emb = get_embeddings(embed_model)
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=emb,
        persist_directory=str(CHROMA_DIR),
    )


def answer_question(
    question: str,
    *,
    embed_model: str | None = None,
    chat_model: str | None = None,
    k: int = 4,
) -> Tuple[str, List[str]]:
    """
    Retrieve top-k chunks and generate an answer. Returns (answer, list of source snippets/files).
    """
    vs = get_vectorstore(embed_model)
    if vs is None:
        return (
            "No documents are indexed yet. Upload files in the sidebar and click **Index uploaded documents**.",
            [],
        )

    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    if not docs:
        return (
            "No relevant passages were found in your documents for this question.",
            [],
        )

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source_file") or d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        src_label = f"{src} (page {page + 1})" if isinstance(page, int) else str(src)
        snippet = d.page_content[:280].replace("\n", " ")
        key = (src_label, snippet)
        if key in seen:
            continue
        seen.add(key)
        sources.append(f"{src_label}: {snippet}...")

    llm = ChatOllama(
        model=chat_model or DEFAULT_CHAT_MODEL,
        base_url=ollama_base_url(),
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm
    msg = chain.invoke({"context": context, "question": question})
    text = msg.content if hasattr(msg, "content") else str(msg)
    return text.strip(), sources
