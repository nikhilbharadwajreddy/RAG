from openai import OpenAI
from pinecone import Pinecone
import os

pinecone_api_key= "pcsk_7DHuPe_6vdFZMwGQBkNFyYJ2iWc5jvUB9Ck9jitrKJAa3DLiJtYj4Ecn4F7raaH4LzJ7D6"
index_host = "https://rag-1k2hyay.svc.aped-4627-b74a.pinecone.io"

openai_api_key = "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"



def query_pinecone(pinecone_api_key,openai_api_key,index_host,question,namespace = "default",model="text-embedding-ada-002"
,top_k=5):
    # Init clients
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(host=index_host)
    openai_client = OpenAI(api_key=openai_api_key)

    # Embed the user query
    query_response = openai_client.embeddings.create(
        model=model,
        input=question
    )
    query_embedding = query_response.data[0].embedding

    # Search Pinecone
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    #Format results
    results = []
    for match in search_results["matches"]:
        metadata = match["metadata"]
        text = metadata.pop("text", "")
        results.append({
            "score": match["score"],
            "chunk": text,
            "metadata": metadata
        })

    return results
