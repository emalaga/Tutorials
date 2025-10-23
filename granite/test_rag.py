import os
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Set environment variable to allow multiple copies of libomp.dylib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "url_loaded" not in st.session_state:
    st.session_state.url_loaded = False

# Initialize models
@st.cache_resource
def get_llm():
    """Initialize the ChatOllama LLM"""
    return ChatOllama(
        model="granite4:micro",
        temperature=0.7,
    )

@st.cache_resource
def get_embeddings():
    """Initialize Ollama embeddings"""
    return OllamaEmbeddings(model="nomic-embed-text")

def load_url_content(url: str):
    """Load and process content from URL"""
    try:
        # Load the URL
        loader = WebBaseLoader(url)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(splits, embeddings)

        return vector_store, len(splits)
    except Exception as e:
        st.error(f"Error loading URL: {str(e)}")
        return None, 0

def create_standalone_question(llm, chat_history, question):
    """
    Reformulate the follow-up question to be standalone by incorporating
    context from chat history
    """
    if not chat_history:
        return question

    # Create a prompt to reformulate the question
    reformulation_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a helpful assistant that reformulates follow-up questions.
Given a chat history and a follow-up question, reformulate the follow-up question to be a
standalone question that contains all necessary context. Do not answer the question, just
reformulate it to be self-contained.

If the question is already standalone and doesn't refer to previous context, return it as is."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Follow-up question: {question}\n\nReformulated standalone question:")
    ])

    # Create the chain
    chain = reformulation_prompt | llm

    # Get the standalone question
    response = chain.invoke({
        "chat_history": chat_history,
        "question": question
    })

    return response.content.strip()

def get_rag_response(vector_store, question, chat_history):
    """Get response from RAG system with chat history"""
    llm = get_llm()

    # First, create a standalone version of the question
    standalone_question = create_standalone_question(llm, chat_history, question)

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Create prompt template
    system_prompt = """You are a helpful assistant answering questions based on the provided context.
Use the following pieces of context to answer the question. If you don't know the answer based on
the context, say so - don't make up information.

Context: {context}

Be conversational and helpful in your responses."""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Get response using the standalone question for retrieval
    # but the original question for the conversation
    response = rag_chain.invoke({
        "input": standalone_question,
        "chat_history": chat_history
    })

    return response["answer"], standalone_question

# Streamlit UI
st.title("RAG Chat Application")
st.caption("Chat with content from any URL using local Ollama models")

# Sidebar for URL input
with st.sidebar:
    st.header("Configuration")

    url_input = st.text_input(
        "Enter URL to load:",
        placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence",
        help="Enter a URL to load and chat about its content"
    )

    if st.button("Load URL", type="primary"):
        if url_input:
            with st.spinner("Loading and processing URL..."):
                vector_store, num_chunks = load_url_content(url_input)
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.url_loaded = True
                    st.session_state.messages = []  # Clear chat history
                    st.success(f"Successfully loaded! Created {num_chunks} chunks.")
        else:
            st.warning("Please enter a URL")

    if st.session_state.url_loaded:
        st.success("✓ Content loaded and ready")

    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### Models Used")
    st.markdown("- **LLM:** granite4:micro")
    st.markdown("- **Embeddings:** nomic-embed-text")
    st.markdown("- **Vector Store:** FAISS (in-memory)")

# Main chat interface
if not st.session_state.url_loaded:
    st.info("👈 Please load a URL from the sidebar to start chatting")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "standalone_question" in message and message["role"] == "assistant":
                with st.expander("View reformulated question"):
                    st.caption(f"Standalone question: {message['standalone_question']}")

    # Chat input
    if prompt := st.chat_input("Ask a question about the content..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Convert messages to LangChain format
                chat_history = []
                for msg in st.session_state.messages[:-1]:  # Exclude the current question
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))

                # Get response
                response, standalone_q = get_rag_response(
                    st.session_state.vector_store,
                    prompt,
                    chat_history
                )

                # Display response
                st.markdown(response)

                # Show the reformulated question
                if standalone_q != prompt:
                    with st.expander("View reformulated question"):
                        st.caption(f"Standalone question: {standalone_q}")

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "standalone_question": standalone_q
        })

        st.rerun()

# Footer
st.divider()
st.caption("Powered by Ollama (granite3.1:2b + nomic-embed-text) and LangChain")
