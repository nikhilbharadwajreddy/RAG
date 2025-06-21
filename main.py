from rag_chunker_pdf import chunk_pdf_with_metadata
from embedding_openai import embed_chunks_with_openai
from save_to_jsonl import save_embeddings_to_jsonl
from pinecone_upsert import upsert_to_pinecone_v3



chunks = chunk_pdf_with_metadata(file_path, doc_id=doc_id, batch_size=20)
embeddings = embed_chunks_with_openai(chunks,key, model="text-embedding-ada-002", batch_size=20)
op = save_embeddings_to_jsonl(embeddings, doc_id, output_dir=".")
upsert_to_pinecone_v3(pinecone_api_key,index_host, op)
