import requests

def embed_chunks_with_openai(chunks, api_key, batch_size=20, model="text-embedding-ada-002"):
    endpoint = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    embedded_chunks = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["chunk"] for c in batch]

        payload = {
            "model": model,
            "input": texts
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Embedding failed: {e}")
            continue

        data = response.json()["data"]
        embeddings = [item["embedding"] for item in data]

        for chunk_obj, embedding in zip(batch, embeddings):
            embedded_chunks.append({
                "id": chunk_obj["metadata"]["chunk_id"],
                "embedding": embedding,
                "metadata": {**chunk_obj["metadata"], "text": chunk_obj["chunk"]}
            })

    return embedded_chunks
