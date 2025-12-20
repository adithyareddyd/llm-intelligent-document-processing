VECTOR_DB = []

def store_vectors(chunks, embeddings):
    for chunk, emb in zip(chunks, embeddings):
        VECTOR_DB.append({
            "text": chunk,
            "embedding": emb
        })

def get_vectors():
    return VECTOR_DB
