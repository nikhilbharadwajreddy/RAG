import os
import json
import sys
import tempfile
from chunker import chunk_document
from embedding_openai import embed_chunks_with_openai
from save_to_jsonl import save_embeddings_to_jsonl
from pinecone_upsert import upsert_to_pinecone_v3


def load_secrets(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

secrets = load_secrets('/Users/bharadwajreddy/Desktop/AI-Projects/sec.json')

openai_key = secrets['openai_key']
pinecone_api_key = secrets['pinecone_api_key']
index_host = secrets['index_host']
# Load secrets

print(secrets)
# Main logic
def process_document(file_path, output_dir="."):
    try:
        print("📄 Chunking document...")
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        chunks = chunk_document(file_path, doc_id=doc_id, batch_size=20)
        print(f"✅ Chunked into {len(chunks)} chunks")

        print("🧠 Generating embeddings...")
        embeddings = embed_chunks_with_openai(chunks, openai_key, model="text-embedding-ada-002", batch_size=20)
        print(f"✅ Got {len(embeddings)} embeddings")

        print("💾 Saving embeddings to JSONL...")
        jsonl_path = save_embeddings_to_jsonl(embeddings, doc_id, output_dir=output_dir)
        print(f"✅ Saved to {jsonl_path}")

        print("⬆️ Upserting to Pinecone...")
        upsert_to_pinecone_v3(pinecone_api_key, index_host, jsonl_path)
        print("✅ Upserted successfully")

        return {
            'success': True,
            'doc_id': doc_id,
            'chunks': len(chunks),
            'embeddings': len(embeddings),
            'jsonl_path': jsonl_path
        }
    except Exception as e:
        print("❌ Error in process_document:", str(e))
        return {'success': False, 'error': str(e)}


filr_path = '/Users/bharadwajreddy/Desktop/resume.docx'

result = process_document(file_path)
