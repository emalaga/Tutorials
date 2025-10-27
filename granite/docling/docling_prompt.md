```
Create a new python script called test.py to create a local RAG based on a pdf. The applications should use the following libraries and models: 
- streamlit for the UI 
- granite4:micro running in a local Ollama instance for the LLM 
- ChatOllama from langchain_ollama 
- DoclingLoader from langchain_docling to parse the pdf 
- langchain_core.prompts and langchain_core.message for Chat functionality 
- sentence-transformers/all-MiniLM-L6-v2 from Hugginface as embedding model 
- langchain_text_splitters to split the text based on the markdown headers
- langchain_milvus for a local Milvus instance to store the vectors
- The RAG application uses as knowledge base a pdf file. The interface should allow to upload a file 

Also the environment variable KMP_DUPLICATE_LIB_OK should be set to TRUE to allow multiple copies of libomp.dylib in memory.
```