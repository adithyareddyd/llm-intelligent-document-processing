\# src/



This folder contains the core backend logic of the Intelligent Document Processing (IDP) system.



\## Modules Overview



\- \*\*pdf\_loader.py\*\*  

&nbsp; Loads and extracts raw text from uploaded PDF documents.



\- \*\*text\_chunker.py\*\*  

&nbsp; Splits large documents into smaller chunks suitable for embeddings and RAG.



\- \*\*embeddings.py\*\*  

&nbsp; Generates vector embeddings using a local LLM embedding model.



\- \*\*vector\_store.py\*\*  

&nbsp; Stores and retrieves document embeddings using FAISS.



\- \*\*retriever.py\*\*  

&nbsp; Retrieves the most relevant document chunks for a given user query.



\- \*\*structured\_extractor.py\*\*  

&nbsp; Extracts structured fields (e.g., name, dates, invoice totals) using LLM prompts.



\- \*\*doc\_type\_detector.py\*\*  

&nbsp; Automatically detects the document type (resume, invoice, report, etc.).



\- \*\*logger.py\*\*  

&nbsp; Logs system actions for audit and debugging purposes.



