import re


GITHUB_PATTERN = r"https://github\.com/[\w\-]+/[\w\-]+"
ARXIV_PATTERN = r"https://(www\.)?arxiv\.org/abs/\d+\.\d+(v\d+)?"


def is_github_url(url):
    """
    Checks whether a URL is a Github URL

    Parameters:
    url (str): URL string to check

    Returns:
    bool: True if it is a Github URL, False otherwise

    >>> is_github_url("https://github.com/openai/gpt-3")
    True
    >>> is_github_url("http://github.com/openai/gpt-3")
    False
    >>> is_github_url("https://openai.com/research")
    False
    """
    return bool(re.match(GITHUB_PATTERN, url))


def is_arxiv_url(url) -> bool:
    """Check if a URL is a valid arXiv URL.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is a valid arXiv URL, False otherwise.
    """
    return bool(re.match(ARXIV_PATTERN, url))


def get_arxiv_id_from_url(url: str) -> str:
    """Extract the arXiv ID from a URL.

    Args:
        url (str): The URL to check.

    Returns:
        str: The arXiv ID if the URL is a valid arXiv URL, else an empty string.
    """
    match = re.match(r"https?://(www\.)?arxiv\.org/abs/(\d+\.\d+(v\d+)?)", url)
    return match.group(2) if match else ""
