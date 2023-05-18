import os
from typing import Any
from langchain.chains.router import MultiPromptChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone

# from consts import INDEX_NAME

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name="manim-doc-index", embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    combined_result = qa({"query": query})
    return combined_result["result"]


if __name__ == "__main__":
    print(
        run_llm(
            query="""produce Python code using Manim CE v0.17.3 ONLY. Then apply a slow zoom 
            towards an object at a specific position in a scene (use a placeholder position, for example [1, 1, 0]). This zoom effect should draw viewers' attention to this position, representing a focus on Zog and his mission. After the zoom, pause for a second. The output should be exclusively Python. 
            If there is any other text other than Python code, just delete it please or COMMENT 
            IT OUT."""
        )
    )
