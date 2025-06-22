from generator import process_document



pinecone_api_key= "pcsk_7DHuPe_6vdFZMwGQBkNFyYJ2iWc5jvUB9Ck9jitrKJAa3DLiJtYj4Ecn4F7raaH4LzJ7D6"
index_host = "https://rag-1k2hyay.svc.aped-4627-b74a.pinecone.io"
file_path = "/Users/bharadwajreddy/Downloads/mid_century_strategy_report-final_red.pdf"  # replace with your actual file
doc_id = "United States Mid-Century Strategy for Deep Decarbonization"
key = "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"

# Process a single document
result = process_document(
    file_path="'/Users/bharadwajreddy/Desktop/resumes/aii/Nikhil Sagili Resume.docx'",
    openai_key=key, 
    pinecone_api_key=pinecone_api_key,
    index_host=index_host,
    output_dir="./embeddings"
)

if result['success']:
    print(f"Processed {result['chunks']} chunks")
else:
    print(f"Error: {result['error']}")