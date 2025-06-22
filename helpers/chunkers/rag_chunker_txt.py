import os
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

def chunk_txt_with_metadata(file_path, doc_id="default_doc", batch_size=20):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
    
    total_lines = len(lines)
    chunk_metadata = []
    chunk_counter = 0

    for i in range(0, total_lines, batch_size):
        batch_text = ""
        line_start = i
        line_end = min(i + batch_size, total_lines) - 1

        for j in range(i, min(i + batch_size, total_lines)):
            line_text = lines[j]
            batch_text += line_text + "\n"

        chunks = text_splitter.split_text(batch_text)

        for idx, chunk_text in enumerate(chunks):
            token_count = len(enc.encode(chunk_text))

            chunk_metadata.append({
                "chunk": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "line_start": line_start,
                    "line_end": line_end,
                    "chunk_id": f"{doc_id}_l{line_start}_c{chunk_counter}",
                    "chunk_number": chunk_counter,
                    "tokens_estimate": token_count
                }
            })

            chunk_counter += 1

    return chunk_metadata
