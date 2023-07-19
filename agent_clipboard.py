import logging
import time
from concurrent.futures import ThreadPoolExecutor

import clipboard
from rich.logging import RichHandler

from agent.tools.embedding_arxiv_paper import embedding_arxiv_paper_from_url
from agent.tools.embedding_local_pdf import embedding_pdf
from agent.tools.rule import (
    is_arxiv_url,
    is_github_url,
    is_wechat_post_url,
)
from agent.tools.summary_of_github import GithubRepoDataCollector
from agent.tools.utils import windows_path_to_wsl
from agent.tools.wechat_post import chain_process

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True)],
)


class ClipboardObserver:
    def update(self, content):
        pass


class ClipboardListener:
    def __init__(self, logger, max_workers=10):
        self._observers: list[ClipboardObserver] = []
        self._last_clipboard_content = None
        self.logger = logger
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown()

    def add_observer(self, observer):
        if not isinstance(observer, ClipboardObserver):
            raise TypeError("The observer must be an instance of ClipboardObserver")
        self._observers.append(observer)

    def remove_observer(self, observer):
        self._observers.remove(observer)

    def _notify_observers(self, content):
        for observer in self._observers:
            try:
                self.executor.submit(observer.update, content)
            except Exception as err:
                self.logger.error(err)

    def listen(self):
        while True:
            try:
                current_clipboard_content = clipboard.paste()

                if current_clipboard_content != self._last_clipboard_content:
                    self._last_clipboard_content = current_clipboard_content
                    self.logger.info("Clipboard content changed")
                    self._notify_observers(current_clipboard_content)

                time.sleep(0.5)
            except Exception as e:
                self.logger.error(e)


class PrintClipboardObserver(ClipboardObserver):
    def __init__(self, logger):
        self.logger = logger

    def update(self, content):
        self.logger.info(f"Clipboard content changed: {content}")


class GitHubLinkObserver(ClipboardObserver):
    def __init__(self, logger):
        self.logger = logger

    def update(self, content):
        if is_github_url(content):
            self.logger.info(f"Detected GitHub link: {content}")
            collector = GithubRepoDataCollector(url=content)
            collector.print_and_save_repo_info()


class ArxivLinkObserver(ClipboardObserver):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def update(self, content):
        if is_arxiv_url(content):
            self.logger.info(f"Detected Arxiv link: {content}")
            for chunk_size in [6000]:
                embedding_arxiv_paper_from_url(
                    url=content, chunk_size=chunk_size, chunk_overlap=0
                )


class LocalPDFObserver(ClipboardObserver):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def update(self, content):
        content = windows_path_to_wsl(content)
        logger.info(content)
        for chunk_size in [6000]:
            embedding_pdf(content, chunk_size=chunk_size, chunk_overlap=0)


class RewriteWechatPostObserver(ClipboardObserver):
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def update(self, content):
        if is_wechat_post_url(content):
            logger.info("Detected Wechat post link: {content}")
            chain_process(content)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    rich_handler = RichHandler(rich_tracebacks=True)
    logger.addHandler(rich_handler)

    with ClipboardListener(logger) as listener:
        listener.add_observer(PrintClipboardObserver(logger))
        listener.add_observer(GitHubLinkObserver(logger))
        listener.add_observer(ArxivLinkObserver(logger))
        listener.add_observer(LocalPDFObserver(logger))
        listener.add_observer(RewriteWechatPostObserver(logger))

        try:
            listener.listen()
        except KeyboardInterrupt:
            logger.info("Stopped listening")
