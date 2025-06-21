from openai import OpenAI
from pinecone import Pinecone
import os


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
