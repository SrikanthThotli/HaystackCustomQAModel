import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os
from pathlib import Path
import shutil
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs
# An in-memory TfidfRetriever based on Pandas dataframes
from haystack.nodes import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader


class BingSearch:

    def __init__(self, subscription_key, url='https://api.bing.microsoft.com/v7.0/search'):
        self.key = subscription_key
        self.url = url
        self.reader_bert = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True)

    def bing_web_search(self, query):
        # set parameters
        search_url = self.url
        headers = {"Ocp-Apim-Subscription-Key": self.key}
        params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
        # get response
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def webpageContentRetriever(self, url):
        html = urlopen(url).read()
        soup = BeautifulSoup(html, features="html.parser")
        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()  # rip it out
        # get text
        text = soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text

    def updateBingFiles(self, search_query, doc_dir):
        # doc_dir = "data/bingfiles/"
        dirpath = Path('data') / 'bingfiles'
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
        try:
            os.makedirs(doc_dir)
            print("Directory '%s' created successfully" % doc_dir)
        except OSError as error:
            print("Directory '%s' can not be created" % doc_dir)
        # call API
        search_results = self.bing_web_search(search_query)
        size = len(search_results['webPages']['value'])

        for ind in range(size):
            try:
                content = self.webpageContentRetriever(search_results['webPages']['value'][ind]['url'])
            except:
                continue
            f = open(doc_dir + str(ind) + "_search_result.txt", "w+", encoding="utf-8")
            f.write(content)
            f.close()

    def getAnswer(self, search_query):

        doc_dir = "data/bingfiles/"
        self.updateBingFiles(search_query, doc_dir)

        # In-Memory Document Store
        document_store_bing = InMemoryDocumentStore()
        docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
        document_store_bing.write_documents(docs)
        # An in-memory TfidfRetriever based on Pandas dataframes
        retriever_bing = TfidfRetriever(document_store=document_store_bing)
        pipe_bing = ExtractiveQAPipeline(self.reader_bert, retriever_bing)
        print("***********************************" + search_query + "****************************************")
        prediction = pipe_bing.run(query=search_query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})

        result_answer = prediction['answers'][0]
        print("********************" + str(result_answer))
        actual_answer = str(result_answer).split("answer='")[1].split("',")[0]
        actual_score = str(result_answer).split("score=")[1].split(",")[0]
        actual_text = str(result_answer).split("context='")[1].split("',")[0]

        result_list = [{'answer': actual_answer, 'score': actual_score, 'text': actual_text}]
        return result_list
