import json
import os
import boto3
import tempfile
from helpers.chunker import chunk_document
from helpers.embedding_openai import embed_chunks_with_openai
from helpers.save_to_jsonl import save_embeddings_to_jsonl
from helpers.pinecone_upsert import upsert_to_pinecone_v3

# Initialize AWS clients
s3_client = boto3.client('s3')
ssm = boto3.client('ssm')

def get_param(name, decrypt=True):
    return ssm.get_parameter(Name=name, WithDecryption=decrypt)['Parameter']['Value']

def download_from_s3(bucket, key, local_path):
    try:
        s3_client.download_file(bucket, key, local_path)
        return True
    except Exception as e:
        print(f"Failed to download {bucket}/{key}: {e}")
        return False

def lambda_handler(event, context):
    try:
        # Get S3 event details
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        # Extract doc_id from filename
        doc_id = os.path.splitext(os.path.basename(key))[0]
        
        print(f"Processing {bucket}/{key} with doc_id: {doc_id}")
        
        # Get API keys from SSM
        openai_key = get_param("openai_api_key", decrypt=True)
        pinecone_api_key = get_param("pinecone_api_key", decrypt=False)
        index_host = get_param("pinecone_index_host", decrypt=True)
        
        if not all([openai_key, pinecone_api_key, index_host]):
            return {
                'statusCode': 500,
                'body': json.dumps('Missing required API keys')
            }
        
        # Download file from S3
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            local_path = tmp_file.name
            
        if not download_from_s3(bucket, key, local_path):
            return {
                'statusCode': 500,
                'body': json.dumps(f'Failed to download {bucket}/{key}')
            }
        
        try:
            # Chunk the document
            chunks = chunk_document(local_path, doc_id=doc_id, batch_size=20)
            print(f"Created {len(chunks)} chunks")
            
            # Create embeddings
            embeddings = embed_chunks_with_openai(
                chunks, 
                openai_key, 
                model="text-embedding-ada-002", 
                batch_size=20
            )
            print(f"Created {len(embeddings)} embeddings")
            
            # Save to JSONL
            jsonl_path = save_embeddings_to_jsonl(embeddings, doc_id, output_dir="/tmp")
            
            # Upload to Pinecone
            upsert_to_pinecone_v3(pinecone_api_key, index_host, jsonl_path)
            
            # Optional: Save JSONL to S3 for backup
            output_key = f"embeddings/{doc_id}.jsonl"
            s3_client.upload_file(jsonl_path, bucket, output_key)
            print(f"Saved embeddings to s3://{bucket}/embeddings/{output_key}")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Document processed successfully',
                    'doc_id': doc_id,
                    'chunks': len(chunks),
                    'embeddings': len(embeddings),
                    'output_s3': f"s3://{bucket}/{output_key}"
                })
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(local_path):
                os.unlink(local_path)
            if os.path.exists(jsonl_path):
                os.unlink(jsonl_path)
                
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }