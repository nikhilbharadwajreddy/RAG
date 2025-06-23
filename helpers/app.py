from flask import Flask, request, jsonify, send_file
import os
import json
import tempfile
from werkzeug.utils import secure_filename
from chunker import chunk_document
from embedding_openai import embed_chunks_with_openai
from save_to_jsonl import save_embeddings_to_jsonl
from pinecone_upsert import upsert_to_pinecone_v3

app = Flask(__name__)

# Load API keys from JSON file
def load_secrets(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Update this path to your JSON file location
secrets = load_secrets('/Users/bharadwajreddy/Desktop/AI-Projects/RAG/sec.json')

OPENAI_KEY = secrets['openai_key']
PINECONE_API_KEY = secrets['pinecone_api_key'] 
INDEX_HOST = secrets['index_host']

@app.route('/')
def index():
    return send_file('upload.html')

def process_document(file_path, openai_key, pinecone_api_key, index_host, output_dir="."):
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


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        result = process_document(tmp.name, OPENAI_KEY, PINECONE_API_KEY, INDEX_HOST)
        os.unlink(tmp.name)
        
    return jsonify(result)

@app.route('/favicon.ico')
def favicon():
    return send_file('favicon.ico')


if __name__ == '__main__':
    app.run(port =5001,debug=True)