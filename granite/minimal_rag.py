# pip install -U langchain-ollama

# rag_min.py
from langchain_ollama import OllamaLLM, OllamaEmbeddings
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

# ---- Tiny "stuff" RAG prompt + local LLM (granite4:micro via Ollama) --------
llm = OllamaLLM(model="granite4:micro")  # local inference

def answer(question: str) -> str:
    # 1) retrieve
    ctx_docs = retriever.invoke(question)
    context = "\n\n".join(f"- {d.metadata.get('title')}: {d.page_content}" for d in ctx_docs)

    # 2) simple prompt (stuff the context)
    prompt = f"""You are a helpful assistant. Use ONLY the context to answer.

Context:
{context}

Question: {question}
Answer (concise):"""
    # 3) generate
    #return llm(prompt)
    return llm.invoke(prompt)

if __name__ == "__main__":
    q = "Who are the two possible fathers of Bridget's baby?"
    print("Q:", q)
    print("A:", answer(q))
