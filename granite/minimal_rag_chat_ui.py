# chat_rag_ui.py
import os

# Allow duplicate OpenMP runtimes on macOS to prevent libomp init crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ----------------- Tiny KB (your JSON) -----------------
DOCUMENTS = [
    {
        "doc_id": 1,
        "title": "Bridget Jones: The Edge of Reason (2004)",
        "text": "Bridget Jones: The Edge of Reason (2004) - Bridget is currently living a happy life with her lawyer boyfriend Mark Darcy, however not only does she start to become threatened and jealous of Mark's new young intern, she is angered by the fact Mark is a Conservative voter. With so many issues already at hand, things get worse for Bridget as her ex-lover, Daniel Cleaver, re-enters her life; the only help she has are her friends and her reliable diary.",
    },
    {
        "doc_id": 2,
        "title": "Bridget Jones's Baby (2016)",
        "text": "Bridget Jones's Baby (2016) - Bridget Jones is struggling with her current state of life, including her break up with her love Mark Darcy. As she pushes forward and works hard to find fulfilment in her life seems to do wonders until she meets a dashing and handsome American named Jack Quant. Things from then on go great, until she discovers that she is pregnant but the biggest twist of all, she does not know if Mark or Jack is the father of her child.",
    },
    {
        "doc_id": 3,
        "title": "Bridget Jones's Diary (2001)",
        "text": "Bridget Jones's Diary (2001) - Bridget Jones is a binge drinking and chain smoking thirty-something British woman trying to keep her love life in order while also dealing with her job as a publisher. When she attends a Christmas party with her parents, they try to set her up with their neighbours' son, Mark. After being snubbed by Mark, she starts to fall for her boss Daniel, a handsome man who begins to send her suggestive e-mails that leads to a dinner date. Daniel reveals that he and Mark attended college together, in that time Mark had an affair with his fiancée. Bridget decides to get a new job as a TV presenter after finding Daniel being frisky with a colleague. At a dinner party, she runs into Mark who expresses his affection for her, Daniel claims he wants Bridget back, the two fight over her and Bridget must make a decision who she wants to be with.",
    },
]

# ----------------- Caching: vector store & models -----------------
@st.cache_resource(show_spinner=False)
def get_retriever(k: int = 2):
    emb = OllamaEmbeddings(model="nomic-embed-text")
    docs = [
        Document(page_content=r["text"], metadata={"title": r["title"], "doc_id": r["doc_id"]})
        for r in DOCUMENTS
    ]
    vs = FAISS.from_documents(docs, emb)
    return vs.as_retriever(search_kwargs={"k": k})

@st.cache_resource(show_spinner=False)
def get_chat(model_name: str = "granite4:micro", temperature: float = 0.2):
    return ChatOllama(model=model_name, temperature=temperature)

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

def to_lc_history(messages):
    """Convert Streamlit history [{'role': 'user'|'assistant','content': str}, ...] to LangChain messages."""
    lc = []
    for m in messages:
        if m["role"] == "user":
            lc.append(HumanMessage(content=m["content"]))
        else:
            lc.append(AIMessage(content=m["content"]))
    return lc

def fmt_context(docs):
    return "\n\n".join(f"- {d.metadata.get('title')}:\n{d.page_content}" for d in docs)

def rag_answer(user_q: str, history, retriever, chat):
    # 1) Condense follow-up
    condense_chain = CONDENSE_PROMPT | chat
    standalone_q = condense_chain.invoke({"chat_history": to_lc_history(history), "question": user_q}).content.strip()

    print(f"[Standalone Question] {standalone_q}")

    # 2) Retrieve
    ctx_docs = retriever.invoke(standalone_q)
    context = fmt_context(ctx_docs)

    # 3) Answer with context + history
    answer_chain = ANSWER_PROMPT | chat
    answer_msg = answer_chain.invoke({
        "chat_history": to_lc_history(history),
        "question": user_q,
        "context": context
    })
    return answer_msg.content.strip(), ctx_docs

# ----------------- UI -----------------
st.set_page_config(page_title="Local RAG (granite4:micro + Streamlit)", page_icon="📚")
st.title("📚 Local RAG Chat — granite4:micro (Ollama)")
st.caption("Minimal conversational RAG over a tiny Bridget Jones dataset. Fully local via Ollama.")

with st.sidebar:
    st.subheader("Settings")
    model = st.text_input("LLM model", value="granite4:micro")
    k = st.number_input("Top-K passages", min_value=1, max_value=5, value=2, step=1)
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    col1, col2 = st.columns(2)
    with col1:
        reset = st.button("Clear chat", use_container_width=True, type="secondary")
    with col2:
        rebuild = st.button("Rebuild index", use_container_width=True)

if "messages" not in st.session_state or reset:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me about Bridget Jones movies!"}]

# Build (or rebuild) resources
retriever = get_retriever(k if not rebuild else 2)  # cache key unaffected by 'rebuild'; keep it simple
chat = get_chat(model, temp)

# Chat transcript
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input box
if user_q := st.chat_input("Type a question…"):
    # Display the user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Generate answer
    try:
        answer, ctx_docs = rag_answer(
            user_q=user_q,
            history=st.session_state.messages,
            retriever=retriever,
            chat=chat
        )
    except Exception as e:
        answer = f"⚠️ Error talking to the local model. Is Ollama running? Details: {e}"
        ctx_docs = []

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
        if ctx_docs:
            with st.expander("Sources"):
                for d in ctx_docs:
                    st.markdown(f"- **[{d.metadata['doc_id']}] {d.metadata['title']}**")

    st.session_state.messages.append({"role": "assistant", "content": answer})
