import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os

# Set Streamlit title and description
st.title("PDF Chatbot with OpenAI")
st.write("Upload a PDF and enter your OpenAI API Key to chat with the PDF.")

# Add a file uploader for the PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Add a text input for the OpenAI API key
openai_api_key = st.text_input("Enter your OpenAI API Key")

def extract_and_chat(pdf_path, openai_key, query):
    os.environ['OPENAI_API_KEY'] = openai_key
    pdfreader = PdfReader(pdf_path)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)

    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    return result

# Add a text input for user questions
user_question = st.text_input("Ask a question")

if st.button("Chat"):
    if uploaded_file is not None and openai_api_key and user_question:
        result = extract_and_chat(uploaded_file, openai_api_key, user_question)
        st.write("Answer:", result)
    else:
        st.warning("Please upload a PDF file, provide an OpenAI API Key, and enter a question.")

if __name__ == "__main__":
    st.set_option('deprecation.showfileUploaderEncoding', False)
