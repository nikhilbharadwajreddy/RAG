import json
import pinecone

# Load secrets
def load_secrets(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

secrets = load_secrets('/Users/bharadwajreddy/Desktop/AI-Projects/sec.json')

openai_key = secrets['openai_key']
pinecone_api_key = secrets['pinecone_api_key']
index_host = secrets['index_host']

# Initialize Pinecone client (v3)
pinecone.init(api_key=pinecone_api_key, host=index_host)


index_name = index_host.split('.')[0]
index = pinecone.Index(index_name)

# Delete all vectors from 'default' namespace
namespace = "default"
print(f"🧹 Deleting all vectors in namespace '{namespace}'...")

index.delete(delete_all=True, namespace=namespace)

print("✅ Deletion complete.")
