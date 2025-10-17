# chat_rag_url_ui.py
import os
import time

os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (compatible; Granite-URL-RAG/1.0)")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ----------------- Streamlit setup -----------------
st.set_page_config(page_title="URL RAG (granite4:micro + Streamlit)", page_icon="🔗")
st.title("🔗 Local RAG Chat from a URL — granite4:micro (Ollama)")
st.caption("Enter a URL (e.g., a Wikipedia page). The app fetches, chunks, embeds, and answers questions locally.")

# ----------------- Sidebar controls -----------------
with st.sidebar:
    st.subheader("Settings")
    default_url = "https://en.wikipedia.org/wiki/Bridget_Jones%27s_Baby"
    url = st.text_input("Source URL", value=default_url, help="Any public web page. Wikipedia works great.").strip()
    k = st.number_input("Top-K passages", min_value=1, max_value=8, value=2, step=1)
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=4000, value=1200, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
    model = st.text_input("LLM model", value="granite4:micro", disabled=True)
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

# ----------------- Chat state priming -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if clear:
    st.session_state.messages = []
    st.session_state.pop("last_loaded_url", None)
    st.session_state.pop("last_retriever_signature", None)

# Reset chat automatically when a new URL arrives
if url:
    if st.session_state.get("last_loaded_url") != url:
        st.session_state.messages = []
        st.session_state["last_loaded_url"] = url
        st.session_state.pop("last_retriever_signature", None)
else:
    st.session_state.pop("last_loaded_url", None)

# ----------------- Caching: model + retriever -----------------
@st.cache_resource(show_spinner=False)
def get_chat(model_name: str, temp: float):
    return ChatOllama(model=model_name, temperature=temp)

@st.cache_resource(show_spinner=True)
def build_retriever(url: str, chunk_size: int, chunk_overlap: int, k: int, nonce: int):
    start_time = time.perf_counter()
    # 1) Load the web page
    loader = WebBaseLoader([url])  # list form for compatibility
    raw_docs = loader.load()
    if not raw_docs:
        raise ValueError("No content was loaded from the URL.")

    # 2) Combine to a single Document for uniform chunking
    combined_text = "\n\n".join(d.page_content for d in raw_docs)
    title = raw_docs[0].metadata.get("title") or url
    base_metadata = {"title": title, "source": url}
    base_doc = Document(page_content=combined_text, metadata=base_metadata)

    # 3) Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents([base_doc])

    # 4) Embed + index in memory
    emb = OllamaEmbeddings(model="nomic-embed-text")
    vs = FAISS.from_documents(chunks, emb)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    elapsed = time.perf_counter() - start_time
    return retriever, elapsed

# Build resources
chat = get_chat(model, temperature)
retriever = None
error_building = None
build_time = None

chunk_size_val = int(chunk_size)
chunk_overlap_val = int(chunk_overlap)
k_val = int(k)

try:
    retriever, build_time = build_retriever(url, chunk_size_val, chunk_overlap_val, k_val, st.session_state.rebuild_nonce)
except Exception as e:
    error_building = str(e)

if build_time is not None and retriever is not None and not error_building:
    signature = "|".join(map(str, [url, chunk_size_val, chunk_overlap_val, k_val, st.session_state.rebuild_nonce]))
    if st.session_state.get("last_retriever_signature") != signature:
        status_msg = (
            f"I have read the source URL: {url}. "
            f"It took me {build_time:.2f} seconds to process. Ask me a question about it."
        )
        st.session_state.messages.append({"role": "assistant", "content": status_msg})
        st.session_state["last_retriever_signature"] = signature

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
    start_time = time.perf_counter()
    lc_history = to_lc_history(history)

    # 1) Condense follow-up
    standalone_q = (CONDENSE_PROMPT | chat).invoke({
        "chat_history": lc_history,
        "question": user_q
    }).content.strip()

    # 2) Retrieve
    ctx_docs = retriever.invoke(standalone_q)
    context = fmt_context(ctx_docs)

    # 3) Answer with context + history
    answer_msg = (ANSWER_PROMPT | chat).invoke({
        "chat_history": lc_history,
        "question": user_q,
        "context": context
    })

    elapsed = time.perf_counter() - start_time
    details = {
        "elapsed": elapsed,
        "original_question": user_q,
        "standalone_question": standalone_q,
    }
    return answer_msg.content.strip(), ctx_docs, details

# Show any build error prominently
if error_building:
    st.error(f"Failed to build the retriever from the URL.\n\nDetails: {error_building}")

# Render chat transcript
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input box (disabled if retriever couldn't be built)
user_q = st.chat_input("Type a question…", disabled=(retriever is None))
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    if retriever is None:
        answer = "⚠️ I couldn't index the URL. Please adjust the URL or try rebuilding the index."
        ctx_docs = []
        details = None
    else:
        try:
            answer, ctx_docs, details = rag_answer(user_q, st.session_state.messages, retriever, chat)
        except Exception as e:
            answer = f"⚠️ Error talking to the local model or retriever. Is Ollama running? Details: {e}"
            ctx_docs = []
            details = None

    with st.chat_message("assistant"):
        st.markdown(answer)
        if ctx_docs:
            with st.expander("Sources"):
                # Show unique sources and titles; most chunks share the same source URL
                seen = set()
                for idx, d in enumerate(ctx_docs, start=1):
                    excerpt = d.page_content.strip()
                    if not excerpt:
                        continue
                    key = hash(excerpt)
                    if key in seen:
                        continue
                    seen.add(key)
                    if len(excerpt) > 600:
                        excerpt = excerpt[:600].rstrip() + "…"
                    ttl = d.metadata.get("title") or d.metadata.get("source") or "Document"
                    st.markdown(f"**Passage {idx} — {ttl}**\n\n{excerpt}")
        if details:
            with st.expander("Details"):
                st.markdown(f"- **Elapsed time:** {details['elapsed']:.2f} seconds")
                st.markdown(f"- **Original question:** {details['original_question']}")
                st.markdown(f"- **Standalone question:** {details['standalone_question']}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
