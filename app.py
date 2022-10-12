import os

import spacy
import streamlit as st
from components import CustomSearch
from haystack.nodes import FARMReader

def welcome():
    return "Welcome all"

@st.cache(allow_output_mutation = True)
def get_model():
    reader_bert = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True)
    return reader_bert

def main():
    st.title ("Custom QA Model about Careers")
    html_temp = """
    <h3>Ask me anything about your career</h3>
    """
    st. markdown(html_temp,unsafe_allow_html=True)
    question = st.text_input("Question","")
    result = ""
    reader_bert = get_model()
    if st.button("Submit"):
        customSearch = CustomSearch()
        result = customSearch.getAnswer(str(question),reader_bert)
    st.success('{}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if __name__ == '__main__':
	main()