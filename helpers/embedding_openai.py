from openai import OpenAI


def embed_chunks_with_openai(chunks,key, batch_size=20, model="text-embedding-ada-002"):
    client = OpenAI(api_key=key)
    embedded_chunks = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["chunk"] for c in batch]

        try:
            response = client.embeddings.create(model=model, input=texts)
        except Exception as e:
            print(f"Embedding failed: {e}")
            continue

        embeddings = [r.embedding for r in response.data]

        for chunk_obj, embedding in zip(batch, embeddings):
            embedded_chunks.append({
                "id": chunk_obj["metadata"]["chunk_id"],
                "embedding": embedding,
                "metadata": {**chunk_obj["metadata"], "text": chunk_obj["chunk"]}
            })

    return embedded_chunks
