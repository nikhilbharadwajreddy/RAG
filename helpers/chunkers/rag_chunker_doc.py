import docx  # python-docx
import mammoth  # mammoth
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

def chunk_doc_with_metadata(file_path, doc_id="default_doc", batch_size=20):
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    total_paragraphs = len(paragraphs)
    chunk_metadata = []
    chunk_counter = 0

    for i in range(0, total_paragraphs, batch_size):
        batch_text = ""
        para_start = i
        para_end = min(i + batch_size, total_paragraphs) - 1

        for j in range(i, min(i + batch_size, total_paragraphs)):
            para_text = paragraphs[j]
            batch_text += para_text + "\n"

        chunks = text_splitter.split_text(batch_text)

        for idx, chunk_text in enumerate(chunks):
            token_count = len(enc.encode(chunk_text))

            chunk_metadata.append({
                "chunk": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "para_start": para_start,
                    "para_end": para_end,
                    "chunk_id": f"{doc_id}_p{para_start}_c{chunk_counter}",
                    "chunk_number": chunk_counter,
                    "tokens_estimate": token_count
                }
            })

            chunk_counter += 1

    return chunk_metadata