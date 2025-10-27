import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_docling import DoclingLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile

# Set environment variable to allow multiple copies of libomp.dylib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# Page configuration
st.set_page_config(
    page_title="PDF RAG with Granite",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Application with Granite")
st.markdown("Upload a PDF and ask questions about its content")

# Sidebar for PDF upload and processing
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None:
        # Check if this is a new file
        if st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # Load and parse PDF with DoclingLoader
                    st.info("Loading PDF with Docling...")
                    loader = DoclingLoader(file_path=tmp_file_path)
                    documents = loader.load()

                    # Split text based on markdown headers
                    st.info("Splitting documents...")
                    headers_to_split_on = [
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3"),
                    ]
                    markdown_splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=headers_to_split_on,
                        strip_headers=False
                    )

                    # Split all documents
                    all_splits = []
                    for doc in documents:
                        splits = markdown_splitter.split_text(doc.page_content)
                        for split in splits:
                            split.metadata.update(doc.metadata)
                            all_splits.append(split)

                    # If no markdown headers found, use the original documents
                    if not all_splits:
                        all_splits = documents

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
                        drop_old=True
                    )

                    st.session_state.processed_file = uploaded_file.name

                    # Clean up temporary file
                    os.unlink(tmp_file_path)

                    st.success(f"✅ Processed {len(all_splits)} document chunks!")

                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
        else:
            st.info(f"✅ Currently using: {uploaded_file.name}")

    if st.session_state.vector_store is not None:
        st.markdown("---")
        st.markdown("### 📊 Status")
        st.success("Vector store ready")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# Main chat interface
if st.session_state.vector_store is None:
    st.info("👈 Please upload a PDF file to get started")
else:
    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message to chat
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant documents
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant context
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )
                    relevant_docs = retriever.invoke(prompt)

                    # Prepare context
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])

                    # Initialize LLM
                    llm = ChatOllama(
                        model="granite4:micro",
                        temperature=0.7
                    )

                    # Create prompt template
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", """You are a helpful assistant that answers questions based on the provided context.
Use the following context to answer the user's question. If you cannot find the answer in the context, say so.

Context:
{context}"""),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{question}")
                    ])

                    # Prepare chat history (last 5 messages for context)
                    chat_history = st.session_state.messages[-6:-1] if len(st.session_state.messages) > 1 else []

                    # Generate response
                    chain = prompt_template | llm
                    response = chain.invoke({
                        "context": context,
                        "chat_history": chat_history,
                        "question": prompt
                    })

                    # Display response
                    st.markdown(response.content)

                    # Add AI response to chat history
                    ai_message = AIMessage(content=response.content)
                    st.session_state.messages.append(ai_message)

                    # Show sources in expander
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(relevant_docs, 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            if doc.metadata:
                                st.caption(f"Metadata: {doc.metadata}")
                            st.markdown("---")

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.info("Make sure Ollama is running with the granite4:micro model")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>Powered by Granite 4 Micro • Langchain • Milvus • Docling</small>
</div>
""", unsafe_allow_html=True)
