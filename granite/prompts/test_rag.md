```
Create a new python script called test_rag.py to create a local RAG that reads a URL (for example a wikipedia article), creates a vector store in memory and then implements a chat experience to answer questions about the article. 

The application should use the following libraries and models: 
- streamlit for the UI 
- granite4:micro running in a local Ollama instance for the LLM 
- ChatOllama from langchain_ollama to connect to the LLM 
- langchain_core.prompts and langchain_core.message for Chat functionality 
-  FAISS from langchain_community.vectorstores for the vector store
- OllamaEmbeddings and the nomic-embed-text model for embeddings

Create a logic to build a standalone version of followup questions to take into consideration past interactions.
Also the environment variable KMP_DUPLICATE_LIB_OK should be set to TRUE to allow multiple copies of libomp.dylib in memory.
```