import math

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    # ---- SAFETY CHECK ----
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)

def retrieve(query_embedding, vector_db, top_k=2):
    scores = []

    for item in vector_db:
        score = cosine_similarity(query_embedding, item["embedding"])
        scores.append((score, item["text"]))

    scores.sort(reverse=True)
    return [text for score, text in scores if score > 0][:top_k]
