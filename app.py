import os

import spacy
import streamlit as st
from components import CustomSearch

def welcome():
    return "Welcome all"

def main():
    st.title ("Custom QA Model about Careers")
    html_temp = """
    <h3>Ask me anything about your career</h3>
    """
    st. markdown(html_temp,unsafe_allow_html=True)
    question = st.text_input("Question","Type your question here...")
    result = ""
    if st.button("Submit"):
        customSearch = CustomSearch()
        result = customSearch.getAnswer(str(question))
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if __name__ == '__main__':
	main()