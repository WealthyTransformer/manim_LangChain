import os
from typing import Any
from langchain import PromptTemplate
from langchain.agents import (
    load_tools,
    initialize_agent,
    Tool,
    AgentType,
)

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.utilities import PythonREPL
import pinecone

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

# Define PythonREPL tool
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

# Load tools
tools = [repl_tool]

# Initialize the agent
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
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
    max_tries = 10  # Maximum number of attempts
    for attempt in range(max_tries):
        try:
            print(f"RetrievalQA attempt {attempt + 1}...")
            result = qa({"query": query})
            print(f"Result from RetrievalQA: {result}, type: {type(result)}")
            print(result)

            code_to_execute = result["result"]  # Get the result using the 'result' key
            print(f"Code to execute: {code_to_execute}")
            prompt_template = PromptTemplate(
                input_variables=["python_code"], template="{python_code}"
            )
            action = agent.run(
                prompt_template.format_prompt(python_code=code_to_execute)
            )  # Use the agent to decide action
            print(f"Action decided by agent: {action}")
            output = repl_tool.func(
                action["tool_input"]
            )  # Execute the action with REPL tool
            print(f"Output from Python REPL: {output}")
            return result  # If no error, return the result
        except Exception as e:
            print(f"Python execution error on attempt {attempt + 1}: {str(e)}")
            if attempt + 1 == max_tries:
                print("Reached maximum number of attempts. Stopping...")
                break
            else:
                print("Retrying RetrievalQAChain...")


if __name__ == "__main__":
    print(
        run_llm(
            query="Write a python code to animate a circle into a Triangle in manim. Use "
            "any values for other inputs that you may require. Make sure you only "
            "ouput Python code and nothing else"
        )
    )
