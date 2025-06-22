import json
import boto3
from openai import OpenAI
from pinecone import Pinecone

# Load from SSM
ssm = boto3.client("ssm")

def get_param(name, decrypt=True):
    return ssm.get_parameter(Name=name, WithDecryption=decrypt)['Parameter']['Value']

def query_pinecone(pinecone_api_key, openai_api_key, index_host, question, namespace="default", model="text-embedding-ada-002", top_k=5):
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(host=index_host)
    openai_client = OpenAI(api_key=openai_api_key)

    query_response = openai_client.embeddings.create(
        model=model,
        input=question
    )
    query_embedding = query_response.data[0].embedding

    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

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

def generate_rag_answer(question, openai_api_key, context_chunks, model="gpt-4"):
    client = OpenAI(api_key=openai_api_key)

    context = "\n\n".join([f"{i+1}. {chunk['chunk']}" for i, chunk in enumerate(context_chunks)])

    prompt = f"""You are a helpful assistant. Use the context below to answer the question along with chunk id in the context from where you got the answer.
If the answer is not contained in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:"""

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return completion.choices[0].message.content.strip()

def lambda_handler(event, context):
    try:
        # Parse request
        body = json.loads(event.get("body", "{}"))
        question = body.get("question")

        if not question:
            return {
                "statusCode": 400,
                "body": "Missing 'question' in request body."
            }

        # Load secrets from SSM
        openai_key = get_param("openai_api_key", decrypt=True)
        pinecone_key = get_param("pinecone_api_key", decrypt=False)
        index_host = get_param("pinecone_index_host", decrypt=True)

        # Query Pinecone
        retrieved_chunks = query_pinecone(
            pinecone_api_key=pinecone_key,
            openai_api_key=openai_key,
            index_host=index_host,
            question=question
        )

        # Generate RAG Answer
        answer = generate_rag_answer(
            question=question,
            openai_api_key=openai_key,
            context_chunks=retrieved_chunks
        )

        return {
            "statusCode": 200,
            "body": answer  
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "body": f"Internal error: {str(e)}"
        }