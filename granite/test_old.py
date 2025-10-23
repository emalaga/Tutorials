import streamlit as st
import tempfile
import os
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_docling import DoclingLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_milvus import Milvus
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Page configuration
st.set_page_config(
    page_title="PDF RAG with Granite",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Application with Granite4:micro")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Sidebar for PDF upload and configuration
with st.sidebar:
    st.header("📄 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload a PDF file to use as knowledge base"
    )

    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                try:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # Load PDF using DoclingLoader
                    st.info("Loading PDF with Docling...")
                    loader = DoclingLoader(file_path=tmp_file_path)
                    documents = loader.load()

                    # Split text by markdown headers
                    st.info("Splitting text by markdown headers...")
                    headers_to_split_on = [
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3"),
                        ("####", "Header 4"),
                    ]

                    markdown_splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=headers_to_split_on,
                        strip_headers=False
                    )

                    # Split all documents
                    all_splits = []
                    for doc in documents:
                        splits = markdown_splitter.split_text(doc.page_content)
                        # Preserve metadata from original document
                        for split in splits:
                            split.metadata.update(doc.metadata)
                        all_splits.extend(splits)

                    st.info(f"Created {len(all_splits)} text chunks")

                    # Initialize embeddings model
                    st.info("Loading embedding model...")
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )

                    # Create Milvus vector store
                    st.info("Creating vector store...")
                    st.session_state.vector_store = Milvus.from_documents(
                        documents=all_splits,
                        embedding=embeddings,
                        connection_args={"uri": "./milvus_demo.db"},
                        drop_old=True,
                        collection_name="pdf_rag_collection"
                    )

                    st.session_state.pdf_processed = True

                    # Clean up temporary file
                    os.unlink(tmp_file_path)

                    st.success(f"✅ PDF processed successfully! {len(all_splits)} chunks indexed.")

                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.exception(e)

    st.divider()

    st.header("⚙️ Configuration")
    st.info("""
    **LLM:** granite4:micro (Ollama)

    **Embeddings:** all-MiniLM-L6-v2

    **Vector Store:** Milvus (local)
    """)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
if not st.session_state.pdf_processed:
    st.info("👈 Please upload and process a PDF document to start chatting")
else:
    # Initialize LLM
    llm = ChatOllama(
        model="granite4:micro",
        temperature=0.7,
    )

    # Create RAG chain
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context from a PDF document.

Context from the document:
{context}

Please answer the user's question based on the context above. If the answer cannot be found in the context, say so clearly."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if question := st.chat_input("Ask a question about your PDF..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # Display user message
        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant documents
                    relevant_docs = retriever.invoke(question)

                    # Prepare context
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])

                    # Prepare chat history
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:  # Exclude current question
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))

                    # Generate response
                    chain = prompt | llm
                    response = chain.invoke({
                        "context": context,
                        "chat_history": chat_history,
                        "question": question
                    })

                    # Display response
                    st.markdown(response.content)

                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.content
                    })

                    # Show sources in expander
                    with st.expander("📄 View Sources"):
                        for i, doc in enumerate(relevant_docs, 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            st.markdown(f"*Metadata: {doc.metadata}*")
                            st.divider()

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.exception(e)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    Powered by Granite4:micro 🪨 | Streamlit | LangChain | Milvus
</div>
""", unsafe_allow_html=True)
