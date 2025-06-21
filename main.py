from rag_chunker_pdf import chunk_pdf_with_metadata
from embedding_openai import embed_chunks_with_openai
from save_to_jsonl import save_embeddings_to_jsonl
from pinecone_upsert import upsert_to_pinecone_v3


pinecone_api_key= "pcsk_7DHuPe_6vdFZMwGQBkNFyYJ2iWc5jvUB9Ck9jitrKJAa3DLiJtYj4Ecn4F7raaH4LzJ7D6"
index_host = "https://rag-1k2hyay.svc.aped-4627-b74a.pinecone.io"
file_path = "/Users/bharadwajreddy/Downloads/mid_century_strategy_report-final_red.pdf"  # replace with your actual file
doc_id = "United States Mid-Century Strategy for Deep Decarbonization"
key = "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"

chunks = chunk_pdf_with_metadata(file_path, doc_id=doc_id, batch_size=20)
embeddings = embed_chunks_with_openai(chunks,key, model="text-embedding-ada-002", batch_size=20)
op = save_embeddings_to_jsonl(embeddings, doc_id, output_dir=".")
upsert_to_pinecone_v3(pinecone_api_key,index_host, op)
