from pathlib import Path

from agent.tools.ingest_doc import ingest_pdf


def embedding_pdf(file: str, chunk_size, chunk_overlap):
    filepath = Path(file)
    if filepath.exists() and filepath.suffix.endswith("pdf"):
        ingest_pdf(filepath, chunk_size, chunk_overlap)
