

from chunker import chunk_document
chunks = chunk_document("/Users/bharadwajreddy/Desktop/resumes/aii/Nikhil Sagili Resume.docx")  





# # Replace with your actual values
# api_key = "pcsk_7DHuPe_6vdFZMwGQBkNFyYJ2iWc5jvUB9Ck9jitrKJAa3DLiJtYj4Ecn4F7raaH4LzJ7D6"  # your Pinecone API key
# index_host = "https://rag-1k2hyay.svc.aped-4627-b74a.pinecone.io"
# index_name = "rag"
# namespace = ""  # Set to "" for default, or provide a namespace if used

# from pinecone import Pinecone



# pc = Pinecone(api_key=api_key)
# index = pc.Index(host=index_host)

# # List and delete all namespaces
# stats = index.describe_index_stats()
# namespaces = stats.get("namespaces", {})
# print("Available namespaces:", namespaces)

# for ns in namespaces.keys():
#     print(f"Deleting vectors in namespace: {ns}")
#     index.delete(delete_all=True, namespace=ns)

