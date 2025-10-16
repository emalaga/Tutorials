
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ---- Your tiny KB (JSON) ----------------------------------------------------
documents = [
    {
        "doc_id": 1,
        "title": "Bridget Jones: The Edge of Reason (2004)",
        "text": "Bridget Jones: The Edge of Reason (2004) - Bridget is currently living a happy life with her lawyer boyfriend Mark Darcy, however not only does she start to become threatened and jealous of Mark's new young intern, she is angered by the fact Mark is a Conservative voter. With so many issues already at hand, things get worse for Bridget as her ex-lover, Daniel Cleaver, re-enters her life; the only help she has are her friends and her reliable diary.",
        "source": ""
    },
    {
        "doc_id": 2,
        "title": "Bridget Jones's Baby (2016)",
        "text": "Bridget Jones's Baby (2016) - Bridget Jones is struggling with her current state of life, including her break up with her love Mark Darcy. As she pushes forward and works hard to find fulfilment in her life seems to do wonders until she meets a dashing and handsome American named Jack Quant. Things from then on go great, until she discovers that she is pregnant but the biggest twist of all, she does not know if Mark or Jack is the father of her child.",
        "source": ""
    },
    {
        "doc_id": 3,
        "title": "Bridget Jones's Diary (2001)",
        "text": "Bridget Jones's Diary (2001) - Bridget Jones is a binge drinking and chain smoking thirty-something British woman trying to keep her love life in order while also dealing with her job as a publisher. When she attends a Christmas party with her parents, they try to set her up with their neighbours' son, Mark. After being snubbed by Mark, she starts to fall for her boss Daniel, a handsome man who begins to send her suggestive e-mails that leads to a dinner date. Daniel reveals that he and Mark attended college together, in that time Mark had an affair with his fiancée. Bridget decides to get a new job as a TV presenter after finding Daniel being frisky with a colleague. At a dinner party, she runs into Mark who expresses his affection for her, Daniel claims he wants Bridget back, the two fight over her and Bridget must make a decision who she wants to be with.",
        "source": ""
    },
]

# ---- Build the store (FAISS + local embeddings via Ollama) -------------------
emb = OllamaEmbeddings(model="nomic-embed-text")
docs = [
    Document(page_content=rec["text"], metadata={"title": rec["title"], "doc_id": rec["doc_id"]})
    for rec in documents
]
vs = FAISS.from_documents(docs, emb)
retriever = vs.as_retriever(search_kwargs={"k": 2})

# --- Local chat model (granite4:micro via Ollama) ----------------------------
chat = ChatOllama(model="granite4:micro", temperature=0.2)

# --- 1) Condense follow-up to standalone question ----------------------------
condense_prompt = ChatPromptTemplate.from_messages([
    ("system", "You turn follow-up questions into standalone questions using the chat history."),
    MessagesPlaceholder("chat_history"),
    ("human", "Rewrite the user question as a standalone question.\nQuestion: {question}")
])
condense_chain = condense_prompt | chat  # returns an AIMessage

# --- 2) Answer with retrieved context + history ------------------------------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use ONLY the provided context to answer. "
     "If the answer is not in the context, say you don't know.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])
answer_chain = answer_prompt | chat  # returns an AIMessage

def format_context(docs):
    return "\n\n".join(f"- {d.metadata.get('title')}:\n{d.page_content}" for d in docs)

def chat_loop():
    print("RAG chat ready. Ask about Bridget Jones. Type 'exit' to quit.\n")
    history = []  # list of HumanMessage/AIMessage

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        # 1) Rephrase to standalone question (uses history)
        standalone_q_msg = condense_chain.invoke({"chat_history": history, "question": user})
        standalone_q = standalone_q_msg.content.strip()

        # 2) Retrieve context based on standalone question
        ctx_docs = retriever.invoke(standalone_q)
        context = format_context(ctx_docs)

        print(f"\n=>History: \n{history}")
        print(f"\n=>Standalone question: {standalone_q}")
        print(f"\n=>Retrieved {len(ctx_docs)} docs:\n{context}\n")

        # 3) Final answer (uses history + context + original question)
        answer_msg = answer_chain.invoke({
            "chat_history": history,
            "question": user,
            "context": context
        })
        answer = answer_msg.content.strip()

        # Show answer
        print(f"\nAssistant: {answer}\n")
        # Show quick sources
        if ctx_docs:
            src = ", ".join(f"[{d.metadata['doc_id']}] {d.metadata['title']}" for d in ctx_docs)
            print(f"Sources: {src}\n")

        # Update history
        history.append(HumanMessage(content=user))
        history.append(AIMessage(content=answer))

if __name__ == "__main__":
    chat_loop()
