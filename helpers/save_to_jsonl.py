import json
import os

def save_embeddings_to_jsonl(embedded_chunks, doc_id, output_dir="."):
    filename = f"{doc_id}.jsonl"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in embedded_chunks:
            json.dump(item, f)
            f.write("\n")
    

    print(f"Saved {len(embedded_chunks)} embeddings to {output_path}")
    return output_path
