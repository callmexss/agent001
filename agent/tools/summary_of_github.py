import json
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.prompts import PromptTemplate
from rich import print as rich_print

TEMPLATE = """
    repo_name:
    repo_description:
    chinese_summary[以下各项请使用列表进行回复]:
        - 摘要:
        - 事实:
        - 底层逻辑:
        - 相关术语和概念:
        - 必备技能:
        - 用波特五力模型分析这个项目:
        - 认知:
        - 逻辑谬误:
        - 批判性思考:
        - 促进思考的问题:
    rate(0-100):
    hash_tags: (
        can be some of [
            #llm, #gpt, #ai, #generate-code, #generate-text,
            #auto-gpt, #docs-qa, #docs-summarize,
        ] and more decide by yourself...
    )
"""

PROMPT_TEMPLATE = """
Here is some info from a github link: "{text}".
Please summarize it using fields in below template: "{template}".
Your response should be as much detail as possible.
And response with a json string which can be parsed by using
json.loads() in python:
"""

CUR = Path(__file__).parent
PROJECT_ROOT = CUR.parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "github_collections"
OUTPUT_DIRECTORY.mkdir(exist_ok=True, parents=True)


class GithubRepoDataCollector:
    def __init__(
        self, url: str, model="gpt-3.5-turbo-16k-0613", temperature=0, logger=None
    ):
        self.url = url
        self.llm = ChatOpenAI(temperature=temperature, model=model)
        self.chain = LLMChain(llm=self.llm, prompt=self.create_prompt(), verbose=True)
        self.logger = logger if logger else logging.getLogger(__name__)

    @staticmethod
    def create_prompt() -> PromptTemplate:
        return PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["text", "template"]
        )

    def get_docs(self) -> str:
        return UnstructuredURLLoader([self.url]).load()

    def get_repo_info(self) -> Dict[str, Any]:
        docs = self.get_docs()
        res = self.chain({"text": docs[:12000], "template": TEMPLATE})
        repo_info = json.loads(res["text"])
        repo_info["repo_url"] = self.url
        repo_info["context"] = docs[0].page_content
        return repo_info

    @staticmethod
    def _get_input(queue: multiprocessing.Queue):
        queue.put(input("File exists. Do you want to overwrite it? (y/n): "))

    def _confirm_overwrite(self, output_directory: str):
        file_path = Path(output_directory) / f"{Path(self.url).name}.json"
        if file_path.exists():
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=self._get_input, args=(queue,))
            process.start()
            process.join(5)
            if process.is_alive():
                print("Timeout. The file will not be overwritten.")
                process.terminate()
                process.join()
                return False
            return queue.get().lower() == "y"
        return True

    def print_and_save_repo_info(self, output_directory: str = OUTPUT_DIRECTORY):
        repo_info = self.get_repo_info()
        rich_print(repo_info)
        file_path = Path(output_directory) / f"{Path(self.url).name}.json"
        if self._confirm_overwrite(file_path):
            self._save_to_file(repo_info, file_path)

    def _save_to_file(self, repo_info: Dict[str, Any], file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(repo_info, f, ensure_ascii=False, indent=4)
