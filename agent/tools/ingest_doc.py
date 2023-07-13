import logging
import pickle
from pathlib import Path

from langchain.document_loaders import PyMuPDFLoader, ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS, Chroma

from agent.settings import PROJECT_ROOT
from agent.tools.utils import generate_safe_filename

VECTOR_DB = PROJECT_ROOT / ".vector_db"
VECTOR_DB.mkdir(exist_ok=True, parents=True)


def ingest_docs(docs_url):
    """Get documents from web pages."""
    loader = ReadTheDocsLoader(docs_url)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open(VECTOR_DB / f"{Path(docs_url).stem}.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


def ingest_pdf(file: Path, chunk_size, chunk_overlap):
    logging.info(f"Ingest for {file}")
    loader = PyMuPDFLoader(file.as_posix())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = loader.load_and_split(splitter)
    embeddings = OpenAIEmbeddings()

    db = Chroma(
        collection_name=generate_safe_filename(file.stem),
        embedding_function=embeddings,
        persist_directory=VECTOR_DB.as_posix(),
    )

    docs_set = set(db.get()["documents"])
    for doc in docs:
        if doc.page_content not in docs_set:
            db.add_documents([doc])
    db.persist()
    del db
