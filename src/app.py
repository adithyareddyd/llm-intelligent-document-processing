from document_manager import save_document
from pdf_loader import load_pdf
from audit_logger import log_action

# --- OPTIONAL Q&A IMPORTS (used only if user says yes) ---
from text_chunker import chunk_text
from embeddings import get_embeddings
from vector_store import store_vectors, get_vectors
from retriever import retrieve

import ollama


# ------------------------
# USER INPUT
# ------------------------
company_id = input("Enter company id (example: company_001): ")
user_name = input("Enter your name: ")
pdf_path = input("Enter PDF path (example: temp.pdf): ")


# ------------------------
# SAVE DOCUMENT
# ------------------------
metadata = save_document(pdf_path, company_id)
print("Document saved successfully")

log_action(user_name, "UPLOAD", metadata["doc_id"])


# ------------------------
# LOAD PDF (OCR handled inside)
# ------------------------
text = load_pdf(metadata["stored_path"])


# ------------------------
# FAST SUMMARY (DEFAULT)
# ------------------------
print("\nGenerating document summary...\n")

summary_response = ollama.chat(
    model="llama3:latest",
    messages=[
        {
            "role": "system",
            "content": "Summarize this document briefly in 3 bullet points."
        },
        {
            "role": "user",
            "content": text[:800]  # keep small for speed
        }
    ]
)

document_summary = summary_response["message"]["content"]

print("📄 DOCUMENT SUMMARY:\n")
print(document_summary)

log_action(user_name, "SUMMARY_VIEW", metadata["doc_id"])


# ------------------------
# ASK USER IF Q&A IS NEEDED
# ------------------------
choice = input("\nDo you want to ask questions about the document? (y/n): ").lower()

if choice != "y":
    print("\nThank you. Summary completed.")
    exit()


# ------------------------
# RAG INDEXING (ONLY IF USER WANTS Q&A)
# ------------------------
print("\nIndexing document for Q&A...\n")

chunks = chunk_text(text)
embeddings = get_embeddings(chunks)
store_vectors(chunks, embeddings)

print("Document indexed for Q&A.")


# ------------------------
# Q&A LOOP (OPTIONAL)
# ------------------------
while True:
    query = input("\nAsk a question about the document (or type exit): ")

    if query.lower() == "exit":
        print("Exiting Q&A.")
        break

    query_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )["embedding"]

    relevant_chunks = retrieve(query_embedding, get_vectors())

    context = "\n".join(relevant_chunks)[:1500]

    # ✅ UPDATED SIMPLE-ENGLISH Q&A PROMPT
    response = ollama.chat(
        model="llama3:instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer using only the document text. "
                    "Use simple and clear English. "
                    "Use short sentences. "
                    "Do not use difficult words. "
                    "If the answer is not in the document, say: "
                    "'This information is not in the document.'"
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ]
    )

    print("\n🧠 ANSWER:\n")
    print(response["message"]["content"])
