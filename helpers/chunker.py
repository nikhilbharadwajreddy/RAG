import os
import mimetypes
from pathlib import Path


from chunkers.rag_chunker_pdf import chunk_pdf_with_metadata
from chunkers.rag_chunker_doc import chunk_doc_with_metadata
from chunkers.rag_chunker_txt import chunk_txt_with_metadata


def chunk_document(file_path, doc_id="default_doc", batch_size=20):
    """
    Route to appropriate chunker based on file extension
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return chunk_pdf_with_metadata(file_path, doc_id, batch_size)
    elif file_ext in ['.doc', '.docx']:
        return chunk_doc_with_metadata(file_path, doc_id, batch_size)
    elif file_ext == '.txt':
        return chunk_txt_with_metadata(file_path, doc_id, batch_size)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")