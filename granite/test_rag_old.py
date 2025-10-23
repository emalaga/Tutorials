import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Initialize embeddings model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize LLM
llm = ChatOllama(model="granite4:micro", temperature=0.7)

# Page configuration
st.set_page_config(page_title="RAG Chat with URL", page_icon="🤖", layout="wide")

st.title("🤖 RAG Chat with URL Content")
st.markdown("Load content from a URL and chat about it using local Ollama models")

# Sidebar for URL input
with st.sidebar:
    st.header("Configuration")
    url_input = st.text_input(
        "Enter URL:",
        placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence",
        help="Enter a URL to load and analyze"
    )

    load_button = st.button("Load URL", type="primary")

    if load_button and url_input:
        with st.spinner("Loading and processing URL content..."):
            try:
                # Load the URL content
                loader = WebBaseLoader(url_input)
                documents = loader.load()

                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                splits = text_splitter.split_documents(documents)

                # Create vector store
                vectorstore = FAISS.from_documents(splits, embeddings)

                # Store in session state
                st.session_state['vectorstore'] = vectorstore
                st.session_state['url_loaded'] = url_input
                st.session_state['num_chunks'] = len(splits)

                st.success(f"✅ Successfully loaded and processed {len(splits)} chunks from the URL!")

            except Exception as e:
                st.error(f"Error loading URL: {str(e)}")

    # Display status
    if 'url_loaded' in st.session_state:
        st.divider()
        st.success("📄 Content Loaded")
        st.write(f"**URL:** {st.session_state['url_loaded']}")
        st.write(f"**Chunks:** {st.session_state['num_chunks']}")

        if st.button("Clear Data"):
            for key in ['vectorstore', 'url_loaded', 'num_chunks', 'chat_history']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Main chat interface
if 'vectorstore' in st.session_state:
    # Display chat history
    for message in st.session_state['chat_history']:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    # Chat input
    user_input = st.chat_input("Ask a question about the content...")

    if user_input:
        # Add user message to chat history
        st.session_state['chat_history'].append(HumanMessage(content=user_input))

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create retriever
                    retriever = st.session_state['vectorstore'].as_retriever(
                        search_kwargs={"k": 4}
                    )

                    # Create prompt template
                    system_prompt = (
                        "You are a helpful assistant that answers questions based on the provided context. "
                        "Use the following pieces of retrieved context to answer the question. "
                        "If you don't know the answer based on the context, say that you don't know. "
                        "Keep your answer concise and relevant.\n\n"
                        "Context:\n{context}"
                    )

                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}")
                    ])

                    # Create chains
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                    # Get response
                    response = rag_chain.invoke({
                        "input": user_input,
                        "chat_history": st.session_state['chat_history'][:-1]  # Exclude current message
                    })

                    answer = response['answer']

                    # Display response
                    st.write(answer)

                    # Add AI message to chat history
                    st.session_state['chat_history'].append(AIMessage(content=answer))

                    # Optionally show retrieved context
                    with st.expander("View Retrieved Context"):
                        for i, doc in enumerate(response['context'], 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(doc.page_content[:300] + "...")
                            st.divider()

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

else:
    # Instructions when no URL is loaded
    st.info("👈 Enter a URL in the sidebar and click 'Load URL' to get started!")

    st.markdown("""
    ### How to use:
    1. Enter a URL in the sidebar (e.g., a Wikipedia article, blog post, or documentation page)
    2. Click **Load URL** to process the content
    3. Start asking questions about the content in the chat below

    ### Features:
    - **Local Processing**: Uses Ollama running locally with granite4:micro model
    - **Vector Search**: Implements FAISS for efficient similarity search
    - **Chat History**: Maintains conversation context for follow-up questions
    - **Retrieved Context**: View the source chunks used to answer your questions

    ### Example URLs to try:
    - https://en.wikipedia.org/wiki/Artificial_intelligence
    - https://en.wikipedia.org/wiki/Machine_learning
    - https://en.wikipedia.org/wiki/Python_(programming_language)
    """)
