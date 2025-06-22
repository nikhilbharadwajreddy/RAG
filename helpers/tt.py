

from chunker import chunk_document
chunks = chunk_document("/Users/bharadwajreddy/Desktop/resumes/aii/Nikhil Sagili Resume.docx")  





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

