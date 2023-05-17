import os
from bs4 import BeautifulSoup
import re
import html

from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class CustomReadTheDocsLoader(ReadTheDocsLoader):
    def __init__(self, paths, encoding=None, errors=None, **kwargs):
        super().__init__(paths, encoding, errors, **kwargs)
        self.file_paths = paths

    def _clean_data(self, text):
        # Remove extra white space
        text = " ".join(text.split())

        # Replace HTML entities with their corresponding characters
        text = html.unescape(text)

        return text

    def load(self):
        documents = []
        hashes = set()
        for path in self.file_paths:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".html"):
                        with open(os.path.join(root, file), "r", errors="ignore") as f:
                            try:
                                data = f.read()
                                soup = BeautifulSoup(data, "lxml")
                                text = self._clean_data(soup.get_text(separator="\n"))
                                hash_value = hash(text)
                                if hash_value in hashes:
                                    print(
                                        f"Duplicated content in file: {os.path.join(root, file)}"
                                    )  # print out when encountering a duplicate
                                    continue
                                hashes.add(hash_value)
                                documents.append(
                                    {
                                        "page_content": text,
                                        "metadata": {
                                            "source": os.path.join(root, file),
                                            "title": soup.title.string
                                            if soup.title
                                            else None,
                                        },
                                    }
                                )
                            except UnicodeDecodeError as e:
                                print(
                                    f"UnicodeDecodeError: {e}. File: {os.path.join(root, file)}"
                                )
        return documents


def ingest_docs() -> None:
    paths = [
        "manim-docs/docs.manim.community/en/stable",
        "flyingframes/flyingframes.readthedocs.io/en/latest",
        "slama",]

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT_REGION"],)

    loader = CustomReadTheDocsLoader(paths)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    documents = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in raw_documents
    ]
    text_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(documents=documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("manim-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name="manim-doc-index")
    print("****** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()
