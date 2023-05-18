import os

# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

if __name__ == "__main__":
    print("hi")
    pdf_path = "/Users/faisal/Desktop/manimVids/manim/manim-docs-select/combined.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100,  separators=["\n\n", "\n", " ", ""])

    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_manim")

    # new_vectorstore = FAISS.load_local("faiss_index_manim", embeddings)
    # qa = RetrievalQA.from_chain_type(
    #     llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    # )

#     res = qa.run(
#         """produce Python code using Manim CE v0.17.3 ONLY. produce Python code using Manim CE v0.17.3 ONLY.
# Make a single class to do all the instructions below:
#
#  The output should be exclusively Python.
#             If there is any other text other than Python code, just delete it please or COMMENT
#             IT OUT."""
#     )
#     print(res)
