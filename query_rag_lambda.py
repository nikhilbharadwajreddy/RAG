import json
import os
import boto3
import requests
from datetime import datetime
from pymongo import MongoClient
from pinecone import Pinecone


# ===== AWS SSM Loader =====
ssm = boto3.client("ssm")



def get_param(name, decrypt=True):
    return ssm.get_parameter(Name=name, WithDecryption=decrypt)['Parameter']['Value']

mongo_uri = get_param("mongo_uri", decrypt=True)

mongo = MongoClient(mongo_uri)
dedup_collection = mongo["slack_events"]["dedup_keys"]
dedup_collection.create_index("createdAt", expireAfterSeconds=600)




# ===== Pinecone Query Logic =====
def query_pinecone(pinecone_api_key, openai_api_key, index_host, question, namespace="default", top_k=5):
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    embedding_url = "https://api.openai.com/v1/embeddings"
    embed_payload = {
        "input": question,
        "model": "text-embedding-ada-002"
    }

    response = requests.post(embedding_url, headers=headers, json=embed_payload)
    embedding = response.json()['data'][0]['embedding']

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(host=index_host)
    search_results = index.query(vector=embedding, top_k=top_k, include_metadata=True, namespace=namespace)

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

# ===== Generate RAG Answer from OpenAI Chat =====
def generate_rag_answer(question, openai_api_key, context_chunks, model="gpt-4"):
    context = "\n\n".join([f"{i+1}. {chunk['chunk']}" for i, chunk in enumerate(context_chunks)])

    prompt = f"""You are a helpful assistant. Use the context below to answer the question along with chunk id in the context from where you got the answer.
If the answer is not contained in the context, say then answer on your own no problem and make the answer looks bit funny

Context:
{context}

Question:
{question}

Answer:"""

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    chat_url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    response = requests.post(chat_url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content'].strip()

def post_message_to_slack(bot_token, channel, text):
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "channel": channel,
        "text": text
    }
    requests.post("https://slack.com/api/chat.postMessage", headers=headers, json=payload)

# ===== Lambda Entry Point =====
def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))

        if body.get("type") == "url_verification":
            return {
                "statusCode": 200,
                "body": body.get("challenge")
            }

        event_id = body.get("event_id")
        event_data = body.get("event", {})
        user_text = event_data.get("text", "").strip()
        channel = event_data.get("channel")

        if event_id and dedup_collection.find_one({"_id": event_id}):
            print(f"Duplicate event {event_id} — skipping.")
            return {"statusCode": 200, "body": "Duplicate event ignored"}

        if event_id:
            dedup_collection.insert_one({"_id": event_id, "createdAt": datetime.utcnow()})

        response = {"statusCode": 200, "body": "OK"}

        if user_text:
            openai_key = get_param("openai_api_key", decrypt=True)
            pinecone_key = get_param("pinecone_api_key", decrypt=False)
            index_host = get_param("pinecone_index_host", decrypt=True)
            slack_token = get_param("slack_bot_token", decrypt=True)
            chunks = query_pinecone(pinecone_key, openai_key, index_host, user_text)
            answer = generate_rag_answer(user_text, openai_key, chunks)

            post_message_to_slack(slack_token, channel, answer)

        return response

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "body": f"Internal error: {str(e)}"
        }
