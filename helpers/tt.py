from generator import process_document


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