# 📄 Intelligent Document Processing (IDP) using LLMs

An enterprise-grade Intelligent Document Processing (IDP) system that uses Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to understand, summarize, and extract information from documents.

---

## 🏗️ System Architecture (High-Level)

1. **User Interface (Streamlit)**
   - Users upload one or more PDF documents.
   - Users can view summaries, extracted data, and ask questions.

2. **Document Ingestion**
   - PDFs are loaded and text is extracted.
   - Large documents are split into smaller chunks.

3. **Embedding & Vector Storage**
   - Each text chunk is converted into vector embeddings.
   - Embeddings are stored in a FAISS vector database.

4. **Retrieval-Augmented Generation (RAG)**
   - User questions are converted into embeddings.
   - Most relevant document chunks are retrieved.
   - Retrieved context is passed to the LLM for accurate answers.

5. **LLM Intelligence Layer**
   - Generates summaries.
   - Answers user questions.
   - Extracts structured fields (e.g., names, totals, dates).

6. **Audit Logging**
   - User actions and system events are logged for traceability.

---

## 🔄 End-to-End Flow

1. Upload PDF → Text extraction  
2. Text chunking → Embedding generation  
3. Store embeddings → FAISS vector store  
4. Ask question → Retrieve relevant chunks  
5. LLM generates accurate, context-aware response  

---

## 🤖 Why RAG is Used

Traditional LLMs may hallucinate or give generic answers.  
This system uses **Retrieval-Augmented Generation (RAG)** to ensure:

- Answers are grounded in document content
- Reduced hallucinations
- Improved accuracy for large documents

---

## 🧩 Key Features

- Multi-PDF upload and analysis
- Context-aware document summarization
- Structured data extraction
- Document type detection
- Multi-language question answering
- Enterprise-style Streamlit dashboard

---

## 🏢 Real-World Use Cases

- **HR**: Resume screening and candidate analysis  
- **Finance**: Invoice and report processing  
- **Legal**: Contract and policy document review  
- **Operations**: SOP and compliance document analysis  

---

## ⚙️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
