import json
import os
from chunker import chunk_document
from embedding_openai import embed_chunks_with_openai
from save_to_jsonl import save_embeddings_to_jsonl
from pinecone_upsert import upsert_to_pinecone_v3

def process_document(file_path, openai_key, pinecone_api_key, index_host, output_dir="."):
    try:
        # Extract doc_id from filename
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"Processing {file_path} with doc_id: {doc_id}")
        
        # Chunk the document
        chunks = chunk_document(file_path, doc_id=doc_id, batch_size=20)
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
        jsonl_path = save_embeddings_to_jsonl(embeddings, doc_id, output_dir=output_dir)
        
        # Upload to Pinecone
        upsert_to_pinecone_v3(pinecone_api_key, index_host, jsonl_path)
        
        return {
            'success': True,
            'message': 'Document processed successfully',
            'doc_id': doc_id,
            'chunks': len(chunks),
            'embeddings': len(embeddings),
            'jsonl_path': jsonl_path
        }
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }




# Process a single document
result = process_document(
    file_path="/Users/bharadwajreddy/Desktop/resumes/aii/Nikhil Sagili Resume.docx",
    openai_key=key, 
    pinecone_api_key=pinecone_api_key,
    index_host=index_host,
    output_dir="/Users/bharadwajreddy/Desktop/AI-Projects/RAG/depricated"
)

if result['success']:
    print(f"Processed {result['chunks']} chunks")
else:
    print(f"Error: {result['error']}")