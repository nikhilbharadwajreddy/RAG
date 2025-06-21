from openai import OpenAI
from pinecone import Pinecone

# from query_vectordb import query_pinecone
# from rag_answer_generator import generate_rag_answer

openai_api_key = "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"
pinecone_api_key= "pcsk_3TMqK3_4ZFsMN38GW4tXVWmY4S5ZiwJWFCbX1h9kkAcQ2TB69RSwieWkqtw3gin79PQgPi"

index_host = "https://rag-1k2hyay.svc.aped-4627-b74a.pinecone.io"

def query_pinecone(pinecone_api_key, openai_api_key, index_host, question,
                   namespace="default", model="text-embedding-ada-002", top_k=5):
    # Init clients
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(host=index_host)
    openai_client = OpenAI(api_key=openai_api_key)

    # Step 1: Embed the query
    query_response = openai_client.embeddings.create(
        model=model,
        input=question
    )
    query_embedding = query_response.data[0].embedding

    # Step 2: Query Pinecone
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    # Step 3: Format results
    results = []
    for match in search_results["matches"]:
        metadata = match.get("metadata", {})
        text = metadata.pop("text", "")
        results.append({
            "score": match["score"],
            "chunk": text,
            "metadata": metadata
        })

    return results

def generate_rag_answer(question, openai_api_key, context_chunks, model="gpt-4"):
    client = OpenAI(api_key=openai_api_key)

    # Step 1: Build context
    context = "\n\n".join([f"{i+1}. {chunk['chunk']}" for i, chunk in enumerate(context_chunks)])

    # Step 2: Prompt construction
    prompt = f"""You are a helpful assistant. Use the context below to answer the question along with chunk id in the context from where you got the answer.
If the answer is not contained in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:"""

    # Step 3: Generate answer
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return completion.choices[0].message.content.strip()


# ✅ Example Usage
if __name__ == "__main__":

    question = "what does The MCS analysis shows reductions in non-CO2 emissions by 2050?"

    # Step 1: Retrieve relevant chunks
    retrieved_chunks = query_pinecone(
        pinecone_api_key=pinecone_api_key,
        openai_api_key=openai_api_key,
        index_host=index_host,
        question=question
    )

    # Print debug info
    print("📄 Retrieved Chunk Numbers:")
    for r in retrieved_chunks:
        print(r["metadata"].get("chunk_number"))

    # Step 2: Generate RAG answer
    answer = generate_rag_answer(
        question=question,
        openai_api_key=openai_api_key,
        context_chunks=retrieved_chunks
    )

    print("\n🧠 Final Answer:")
    print(answer)
