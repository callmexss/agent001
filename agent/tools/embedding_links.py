import re
from datetime import datetime
from pathlib import Path

from langchain.document_loaders import BSHTMLLoader, UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from agent import settings

RULES = [
    
]


def match_url_with_regex_only(url):
    for rule in RULES:
        if re.search(rule, url):
            return True
    return False


def ingest_urls(urls, db_name=None):
    if len(urls) == 1:
        path = Path(urls[0])
        db_name = path.name
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%s")
        db_name = (db_name if db_name else "urls") + f"_{timestamp}"
    data = UnstructuredURLLoader(urls)

    docs = data.load_and_split(
        CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    )

    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    db.save_local(f"{settings.FAISS_PATH}/{db_name}")
    return db


if __name__ == "__main__":
    url = 'https://weaviate.io/blog/what-is-a-vector-database'

    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0613",
        streaming=True,
        temperature=0,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    db = ingest_urls([url])

    chain = RetrievalQA.from_llm(
        llm=llm,
        verbose=True,
        retriever=db.as_retriever(),
        return_source_documents=True,
    )

    chain1 = RetrievalQAWithSourcesChain.from_llm(
        llm,
        retriever=db.as_retriever(),
    )

    res = chain("什么是向量数据库？")
    res = chain("向量数据库和文本搜索相比的优劣势有哪些？")
    res = chain("有时候向量数据库的召回率非常低，这种时候该如何提升?")
    res = chain("比较短的文本，比如一句话，向量数据库的效果如何？")
    res = chain("如何衡量向量数据库的性能？")

    res = chain1("什么是向量数据库？")
    res = chain1("向量数据库和文本搜索相比的优劣势有哪些？")
    res = chain1("有时候向量数据库的召回率非常低，这种时候该如何提升?")
    res = chain1("比较短的文本，比如一句话，向量数据库的效果如何？")
    res = chain1("如何衡量向量数据库的性能？")

    urls = [
        'https://python.langchain.com/docs/use_cases/question_answering/',
        'https://python.langchain.com/docs/use_cases/question_answering/how_to/document-context-aware-QA',
        'https://python.langchain.com/docs/modules/data_connection/retrievers/',
        'https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.svm.SVMRetriever.html',
        'https://python.langchain.com/docs/integrations/document_loaders/grobid',
        'https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever',
        'https://python.langchain.com/docs/modules/chains/foundational/router',
    ]
    db = ingest_urls(urls, "langchain")
    db.similarity_search_with_score("MMR是什么意思？")

    chain = RetrievalQA.from_llm(
        llm=llm,
        verbose=True,
        retriever=db.as_retriever(),
        return_source_documents=True,
    )

    chain1 = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
    )

    chain2 = RetrievalQAWithSourcesChain.from_llm(
        llm,
        retriever=db.as_retriever(),
    )
    chain3 = RetrievalQAWithSourcesChain.from_chain_type(
        llm,
        retriever=db.as_retriever(),
    )

    db.search("MMR是什么意思？", search_type="similarity")
    db.search("MMR是什么意思？", search_type="mmr")

    res = chain("什么是langchain？")
    res = chain1("什么是langchain？")
    res = chain2("什么是langchain？")
    res = chain2("什么是router？")
    res = chain2("如何使用router？")
    res = chain("什么是向量搜索的 MMR？")
    res = chain1("什么是向量搜索的 MMR？")
    res = chain2("什么是向量搜索的 MMR？")
    res = chain3("什么是向量搜索的 MMR？")
    res = chain2("如何提高问答系统的回答质量？")


    import logging

    from langchain.chat_models import ChatOpenAI
    from langchain.retrievers.multi_query import MultiQueryRetriever

    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=ChatOpenAI(temperature=0),
    )
    question = "什么是mmr？"
    docs = db.similarity_search(question)
    len(docs)
    unique_docs = retriever_from_llm.get_relevant_documents(query=question)
    len(unique_docs)

    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    qa_chain({"query": question})

    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": question})
    result["result"]

    from langchain.chains.question_answering import load_qa_chain

    chain = load_qa_chain(llm, chain_type="stuff")
    chain({"input_documents": unique_docs, "question": question}, return_only_outputs=True)

    from langchain.chains.router import MultiPromptChain
    from langchain.chains import ConversationChain
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate

    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise and easy to understand manner. \
    When you don't know the answer to a question you admit that you don't know.

    Here is a question:
    {input}"""


    math_template = """You are a very good mathematician. You are great at answering math questions. \
    You are so good because you are able to break down hard problems into their component parts, \
    answer the component parts, and then put them together to answer the broader question.

    Here is a question:
    {input}"""

    prompt_infos = [
        {
            "name": "physics",
            "description": "Good for answering questions about physics",
            "prompt_template": physics_template,
        },
        {
            "name": "math",
            "description": "Good for answering math questions",
            "prompt_template": math_template,
        },
    ]

    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain
    default_chain = ConversationChain(llm=llm, output_key="text")

    from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
    from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )

    print(chain.run("What is black body radiation?"))
    print(chain.run("什么是波粒二象性？"))
    print(
        chain.run(
            "What is the first prime number greater than 40 such that one plus the prime number is divisible by 3?"
        )
    )
    print(chain.run("What is the name of the type of cloud that rains?"))

    from langchain.chains.router.embedding_router import EmbeddingRouterChain
    from langchain.embeddings import CohereEmbeddings
    from langchain.vectorstores import Chroma

    names_and_descriptions = [
        ("physics", ["for questions about physics"]),
        ("math", ["for questions about math"]),
    ]

    router_chain = EmbeddingRouterChain.from_names_and_descriptions(
        names_and_descriptions, Chroma, OpenAIEmbeddings(), routing_keys=["input"]
    )

    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )

    print(chain.run("What is black body radiation?"))

    print(
        chain.run(
            "What is the first prime number greater than 40 such that one plus the prime number is divisible by 3?"
        )
    )
