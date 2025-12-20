import ollama

def summarize(text):
    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes documents clearly."
            },
            {
                "role": "user",
                "content": text[:3000]
            }
        ]
    )

    return response["message"]["content"]
