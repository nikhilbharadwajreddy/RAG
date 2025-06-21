import fitz  # PyMuPDF
from langchain.text_splitter import TokenTextSplitter
import tiktoken

# Token splitter setup
text_splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    encoding_name="cl100k_base"
)

# Accurate tokenizer
enc = tiktoken.get_encoding("cl100k_base")

def chunk_pdf_with_metadata(file_path, doc_id="default_doc", batch_size=20):
    doc = fitz.open(file_path)
    total_pages = len(doc)
    chunk_metadata = []
    chunk_counter = 0

    for i in range(0, total_pages, batch_size):
        batch_text = ""
        page_start = i
        page_end = min(i + batch_size, total_pages) - 1  # avoid overflow

        for j, page in enumerate(doc[i:i+batch_size]):
            page_text = page.get_text()
            batch_text += page_text + "\n"

        chunks = text_splitter.split_text(batch_text)

        for idx, chunk_text in enumerate(chunks):
            token_count = len(enc.encode(chunk_text))  # precise token count

            chunk_metadata.append({
                "chunk": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_id": f"{doc_id}_p{page_start}_c{chunk_counter}",
                    "chunk_number": chunk_counter,
                    "tokens_estimate": token_count
                }
            })

            chunk_counter += 1

    return chunk_metadata


