

import json
from pinecone import Pinecone, ServerlessSpec

def load_embeddings_from_jsonl(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def upsert_to_pinecone_v3( api_key, index_host,jsonl_path,namespace = "default"):
    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=index_host)

    items = load_embeddings_from_jsonl(jsonl_path)

    # You must upsert dicts like: {id, values, metadata}
    records = []
    for item in items:
        records.append({
            "id": item["id"],
            "values": item["embedding"],
            "metadata": item["metadata"]
        })

    print(f"⏫ Upserting {len(records)} records into namespace '{namespace}'...")
    index.upsert(vectors=records, namespace=namespace)
    print("✅ Upsert complete.")
