# chat_rag_docling_v2_ui.py
import hashlib
import os
from pathlib import Path
from tempfile import mkdtemp, NamedTemporaryFile

import streamlit as st

from langchain_docling import DoclingLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# ----------------- Streamlit setup -----------------
st.set_page_config(page_title="PDF RAG (Docling + Milvus)", page_icon="📄")
st.title("📄 Local RAG Chat from a PDF — Docling + Milvus + Granite")
st.caption("Upload a PDF. The app parses it with Docling, stores header-aware chunks in a local Milvus DB, and answers questions with granite4:micro.")

# ----------------- Sidebar controls -----------------
with st.sidebar:
    st.subheader("Settings")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    k = st.number_input("Top-K passages", min_value=1, max_value=8, value=3, step=1)
    model = st.text_input("LLM model", value="granite4:micro")
    embedding_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2").strip() or "sentence-transformers/all-MiniLM-L6-v2"
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    col1, col2 = st.columns(2)
    with col1:
        clear = st.button("Clear chat", use_container_width=True, type="secondary")
    with col2:
        rebuild = st.button("Rebuild index", use_container_width=True)

# Track a nonce to force cache refresh when user clicks "Rebuild index"
if "rebuild_nonce" not in st.session_state:
    st.session_state.rebuild_nonce = 0
if rebuild:
    st.session_state.rebuild_nonce += 1


# ----------------- Caching: model + retriever -----------------
@st.cache_resource(show_spinner=False)
def get_chat(model_name: str, temp: float):
    return ChatOllama(model=model_name, temperature=temp)


@st.cache_resource(show_spinner=True)
def build_retriever_from_pdfbytes(file_hash: str, file_name: str, raw_bytes: bytes, k: int,
                                  embedding_model_id: str, nonce: int):
    # Persist the upload so Docling can read it
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        loader = DoclingLoader(tmp_path)
        docs = loader.load()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not docs:
        raise ValueError("Docling returned no content from the PDF.")

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ],
    )
    splits = []
    for doc in docs:
        pieces = splitter.split_text(doc.page_content)
        for split in pieces:
            src = doc.metadata.get("source", file_name or "uploaded-pdf")
            ttl = doc.metadata.get("title", file_name or "Uploaded PDF")
            split.metadata.setdefault("source", src)
            split.metadata.setdefault("title", ttl)
            splits.append(split)

    if not splits:
        raise ValueError("The Markdown header splitter produced no chunks.")

    embedding = HuggingFaceEmbeddings(model_name=embedding_model_id)

    temp_dir = mkdtemp(prefix=f"docling_{nonce}_")
    milvus_uri = str(Path(temp_dir) / "docling.db")
    vectorstore = Milvus.from_documents(
        documents=splits,
        embedding=embedding,
        collection_name="docling_demo",
        connection_args={"uri": milvus_uri},
        index_params={"index_type": "FLAT"},
        drop_old=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


# Build resources if a file is present
chat = get_chat(model, temperature)
retriever = None
error_building = None
file_hash = None
display_name = None

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    hasher = hashlib.md5()
    hasher.update(uploaded_bytes)
    file_hash = hasher.hexdigest()
    display_name = uploaded_file.name
    try:
        retriever = build_retriever_from_pdfbytes(
            file_hash=file_hash,
            file_name=display_name,
            raw_bytes=uploaded_bytes,
            k=int(k),
            embedding_model_id=embedding_model,
            nonce=st.session_state.rebuild_nonce
        )
    except Exception as e:
        error_building = str(e)


# ----------------- Prompts -----------------
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You turn follow-up questions into standalone questions using the chat history."),
    MessagesPlaceholder("chat_history"),
    ("human", "Rewrite the user question as a standalone question.\nQuestion: {question}")
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use ONLY the provided context to answer. "
     "If the answer is not in the context, say you don't know.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])


# ----------------- Helpers -----------------
def to_lc_history(messages):
    lc = []
    for m in messages:
        if m["role"] == "user":
            lc.append(HumanMessage(content=m["content"]))
        else:
            lc.append(AIMessage(content=m["content"]))
    return lc


def fmt_context(docs):
    return "\n\n".join(f"- {d.metadata.get('title', d.metadata.get('source', 'Document'))}:\n{d.page_content}"
                       for d in docs)


def rag_answer(user_q: str, history, retriever, chat):
    # 1) Condense follow-up
    standalone_q = (CONDENSE_PROMPT | chat).invoke({
        "chat_history": to_lc_history(history),
        "question": user_q
    }).content.strip()

    # 2) Retrieve
    ctx_docs = retriever.invoke(standalone_q)
    context = fmt_context(ctx_docs)

    # 3) Answer with context + history
    answer_msg = (ANSWER_PROMPT | chat).invoke({
        "chat_history": to_lc_history(history),
        "question": user_q,
        "context": context
    })
    return answer_msg.content.strip(), ctx_docs


# ----------------- Chat state -----------------
default_greeting = "Upload a PDF on the left and ask me anything about it!"
if "messages" not in st.session_state or clear:
    st.session_state.messages = [{"role": "assistant", "content": default_greeting}]

# Show any build error prominently
if error_building:
    st.error(f"Failed to build the retriever from the PDF.\n\nDetails: {error_building}")

# If no PDF yet, nudge the user and show existing transcript
if uploaded_file is None:
    st.info("⬆️ Upload a PDF to begin. The app expects text-based PDFs; for scans, run OCR first.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input(
        f"Ask about: {display_name}" if display_name else "Type a question…",
        disabled=(retriever is None)
    )
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        if retriever is None:
            answer = "⚠️ I couldn't index the PDF. Please verify the file is text-based and try rebuilding the index."
            ctx_docs = []
        else:
            try:
                answer, ctx_docs = rag_answer(user_q, st.session_state.messages, retriever, chat)
            except Exception as e:
                answer = f"⚠️ Error talking to the local model or retriever. Is Ollama running? Details: {e}"
                ctx_docs = []

        with st.chat_message("assistant"):
            st.markdown(answer)
            if ctx_docs:
                with st.expander("Sources"):
                    seen = set()
                    for d in ctx_docs:
                        ttl = d.metadata.get("title", display_name or "Uploaded PDF")
                        src = d.metadata.get("source", display_name or "uploaded-pdf")
                        key = (ttl, src)
                        if key in seen:
                            continue
                        seen.add(key)
                        st.markdown(f"- **{ttl}** — _{src}_")

        st.session_state.messages.append({"role": "assistant", "content": answer})
