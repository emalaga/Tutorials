# chat_rag_pdf_ui.py
import os, io, tempfile, hashlib
import streamlit as st

from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# - to avoid: OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ----------------- Streamlit setup -----------------
st.set_page_config(page_title="PDF RAG (granite4:micro + Docling)", page_icon="📄")
st.title("📄 Local RAG Chat from a PDF — granite4:micro (Ollama)")
st.caption("Upload a PDF. The app parses it with Docling, chunks, embeds locally, and answers questions with granite4:micro.")

# ----------------- Sidebar controls -----------------
with st.sidebar:
    st.subheader("Settings")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    k = st.number_input("Top-K passages", min_value=1, max_value=8, value=3, step=1)
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=4000, value=1200, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200, step=50)
    model = st.text_input("LLM model", value="granite4:micro")
    embedding_model = st.text_input("Embedding model", value="nomic-embed-text")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    embedding_model = embedding_model.strip() or "nomic-embed-text"

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

# ----------------- Helpers -----------------
def _hash_bytes(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()

def _save_to_tmp(pdf_file) -> str:
    """Persist uploaded PDF to a temp path and return the path."""
    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(pdf_file.read())
        return tmp.name

def _docling_to_text(pdf_path: str) -> str:
    """Parse PDF with Docling and return markdown/plain text."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    # Prefer markdown export; fall back to plain text if needed.
    text = None
    for attr in ("export_to_markdown", "export_to_text"):
        if hasattr(doc, attr):
            try:
                text = getattr(doc, attr)()
                break
            except Exception:
                pass
    if not text:
        # Last resort: join page texts if available
        try:
            text = "\n\n".join(p.text for p in doc.pages)  # may vary by docling version
        except Exception:
            text = ""
    return text or ""

def _make_chunks(text: str, title: str, source_tag: str, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
    base_doc = Document(page_content=text, metadata={"title": title, "source": source_tag})
    return splitter.split_documents([base_doc])

# ----------------- Caching: model + retriever -----------------
@st.cache_resource(show_spinner=False)
def get_chat(model_name: str, temp: float):
    return ChatOllama(model=model_name, temperature=temp)

@st.cache_resource(show_spinner=True)
def build_retriever_from_pdfbytes(file_hash: str, file_name: str, raw_bytes: bytes,
                                  chunk_size: int, chunk_overlap: int, k: int,
                                  embedding_model: str, nonce: int):
    # Persist to a temp file for Docling
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    # Parse with Docling
    text = _docling_to_text(tmp_path)
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    if not text.strip():
        raise ValueError("Docling produced empty text. The PDF may be image-only or encrypted.")

    # Chunk
    chunks = _make_chunks(text=text, title=file_name or "Uploaded PDF", source_tag=file_name or "local-pdf",
                          chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Embed + in-memory FAISS
    try:
        emb = OllamaEmbeddings(model=embedding_model)
        vs = FAISS.from_documents(chunks, emb)
    except Exception as exc:
        raise RuntimeError(
            "Failed to embed chunks with Ollama. "
            "Confirm the embedding model is pulled and that `ollama serve` is running."
        ) from exc
    return vs.as_retriever(search_kwargs={"k": k})

# Build resources if a file is present
chat = get_chat(model, temperature)
retriever = None
error_building = None
file_hash = None
display_name = None

if uploaded_file is not None:
    # Read once and hash to key the cache
    uploaded_bytes = uploaded_file.getvalue()
    file_hash = _hash_bytes(uploaded_bytes)
    display_name = uploaded_file.name
    try:
        retriever = build_retriever_from_pdfbytes(
            file_hash=file_hash,
            file_name=display_name,
            raw_bytes=uploaded_bytes,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            k=int(k),
            embedding_model=embedding_model,
            nonce=st.session_state.rebuild_nonce
        )
    except Exception as e:
        error_building = str(e)

# ----------------- Prompts (same as URL version) -----------------
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
if "messages" not in st.session_state or clear:
    st.session_state.messages = [{"role": "assistant", "content": "Upload a PDF on the left and ask me anything about it!"}]

# Show any build error prominently
if error_building:
    st.error(f"Failed to build the retriever from the PDF.\n\nDetails: {error_building}")

# If no PDF yet, nudge the user
if uploaded_file is None:
    st.info("⬆️ Upload a PDF to begin. Supports text-based PDFs; for pure scans, consider OCR first.")
    # Render existing transcript (e.g., after Clear chat)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
else:
    # Render transcript
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input box (disabled if retriever couldn't be built)
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
                    shown = set()
                    for d in ctx_docs:
                        ttl = d.metadata.get("title", display_name or "Uploaded PDF")
                        src = d.metadata.get("source", display_name or "local-pdf")
                        key = (ttl, src)
                        if key in shown:
                            continue
                        shown.add(key)
                        st.markdown(f"- **{ttl}** — _{src}_")

        st.session_state.messages.append({"role": "assistant", "content": answer})
