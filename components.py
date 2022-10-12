import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import os
from pathlib import Path
import shutil
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs
# An in-memory TfidfRetriever based on Pandas dataframes
from haystack.nodes import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline


class CustomSearch:

    def updateDataFiles(self, doc_dir):
        dirpath = Path('data') / 'files'
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
        try:
            os.makedirs(doc_dir)
            print("Directory '%s' created successfully" % doc_dir)
        except OSError as error:
            print("Directory '%s' can not be created" % doc_dir)

        # Here we need to upload the data corpus of questions and answers which is present in csv file
        # Here we have taken a limited set of questions and answers(~around 1000) to avoid memory issues in colab
        data_file = pd.read_csv("data/CareerVillageDataSet.csv")
        data_file['questions'] = data_file['questions'].fillna("No Question")
        data_file['answers'] = data_file['answers'].fillna("No Answer")
        data_file = data_file.astype(str)
            
        for ind in data_file.index:
            doc_name = ''.join(e for e in data_file['questions'][ind] if e.isalnum())
            f = open(doc_dir + str(ind) + "_" + doc_name + ".txt", "w+", encoding="utf-8")
            f.write(data_file['answers'][ind])
            f.close()

    def getAnswer(self, search_query, reader_bert):

        doc_dir = "data/files/"
        self.updateDataFiles(doc_dir)

        # In-Memory Document Store
        document_store_custom = InMemoryDocumentStore()
        docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
        document_store_custom.write_documents(docs)
        # An in-memory TfidfRetriever based on Pandas dataframes
        retriever_custom = TfidfRetriever(document_store=document_store_custom)
        pipe_custom = ExtractiveQAPipeline(reader_bert, retriever_custom)
        print("***********************************" + search_query + "****************************************")
        prediction = pipe_custom.run(query=search_query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})

        result_answer = prediction['answers'][0]
        print("********************" + str(result_answer))
        actual_answer = str(result_answer).split("answer='")[1].split("',")[0]
        actual_score = str(result_answer).split("score=")[1].split(",")[0]
        actual_text = str(result_answer).split("context='")[1].split("',")[0]

        result_list = [{'answer': actual_answer, 'score': actual_score, 'text': actual_text}]
        return actual_answer
