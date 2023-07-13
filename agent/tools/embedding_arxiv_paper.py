import logging
from functools import wraps
from pathlib import Path

import arxiv

from agent.settings import PROJECT_ROOT
from agent.tools.ingest_doc import ingest_pdf
from agent.tools.rule import get_arxiv_id_from_url
from agent.tools.utils import generate_safe_filename

logger = logging.getLogger(__name__)

DATA_PAPER = PROJECT_ROOT / "data/papers"
DATA_PAPER.mkdir(exist_ok=True, parents=True)

DOWNLOADED_PAPERS_FILE = DATA_PAPER / ".downloaded_papers"


def manage_downloaded_papers(func):
    @wraps(func)
    def wrapper(paper_id: str, *args, **kwargs):
        downloaded_papers = _load_downloaded_papers()

        if paper_id in downloaded_papers:
            logger.info(f"Paper with ID: {paper_id} has already been downloaded")
            paper = next(arxiv.Search(id_list=[paper_id]).results())
            filename = generate_safe_filename(paper.title) + ".pdf"
            return Path(DATA_PAPER) / filename

        result = func(paper_id, *args, **kwargs)
        downloaded_papers.add(paper_id)
        _save_downloaded_papers(downloaded_papers)
        return result

    return wrapper


def _load_downloaded_papers():
    if DOWNLOADED_PAPERS_FILE.exists():
        with open(DOWNLOADED_PAPERS_FILE, "r") as file:
            return set(line.strip() for line in file)
    else:
        return set()


def _save_downloaded_papers(downloaded_papers):
    with open(DOWNLOADED_PAPERS_FILE, "w") as file:
        for paper_id in downloaded_papers:
            file.write(f"{paper_id}\n")


@manage_downloaded_papers
def download_pdf(paper_id: str, dirpath: str = DATA_PAPER):
    logger.info(f"Starting download of paper with ID: {paper_id}")
    paper = next(arxiv.Search(id_list=[paper_id]).results())
    filename = generate_safe_filename(paper.title)
    paper.download_pdf(dirpath=dirpath, filename=filename + ".pdf")
    logger.info(f"Downloaded paper with ID: {paper_id}")
    return Path(dirpath) / (filename + ".pdf")


def embedding_pdf(file: Path, chunk_size, chunk_overlap):
    logger.info(f"Starting embedding of PDF file: {file}")
    ingest_pdf(file, chunk_size, chunk_overlap)
    logger.info(f"Embedded PDF file: {file}")


def embedding_arxiv_paper(paper_id: str, chunk_size, chunk_overlap):
    file = download_pdf(paper_id)
    ingest_pdf(file, chunk_size, chunk_overlap)


def embedding_arxiv_paper_from_url(url: str, chunk_size, chunk_overlap):
    logger.info(f"Received request to embed paper from URL: {url}")
    paper_id = get_arxiv_id_from_url(url)
    embedding_arxiv_paper(paper_id, chunk_size, chunk_overlap)
