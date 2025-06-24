import json
from pinecone import Pinecone

def load_secrets(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def delete_all_vectors(namespace="default"):
    secrets = load_secrets('/Users/bharadwajreddy/Desktop/AI-Projects/sec.json')

    openai_key = secrets['openai_key']
    pinecone_api_key = secrets['pinecone_api_key']
    index_host = secrets['index_host']

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(host=index_host)

    print(f" Deleting all vectors in namespace '{namespace}'...")
    index.delete(delete_all=True, namespace=namespace)
    print("Deletion complete.")


if __name__ == "__main__":
    delete_all_vectors()
