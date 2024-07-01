from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
)
from llama_index.core import SummaryIndex
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.openai import OpenAI

wiki_titles = [
    "Toronto",
    "Seattle",
    "Chicago",
    "Boston",
    "Houston",
]


from pathlib import Path

import requests

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)


# Load all wiki documents
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()


llm = OpenAI(temperature=0, model="gpt-4-turbo")
callback_manager = CallbackManager([])


from llama_index.agent.openai import OpenAIAgent
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter
import os

node_parser = SentenceSplitter()

# Build agents dictionary
query_engine_tools = []

for idx, wiki_title in enumerate(wiki_titles):
    nodes = node_parser.get_nodes_from_documents(city_docs[wiki_title])

    if not os.path.exists(f"./data/{wiki_title}"):
        # build vector index
        vector_index = VectorStoreIndex(
            nodes, callback_manager=callback_manager
        )
        vector_index.storage_context.persist(
            persist_dir=f"./data/{wiki_title}"
        )
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./data/{wiki_title}"),
            callback_manager=callback_manager,
        )
    # define query engines
    vector_query_engine = vector_index.as_query_engine(llm=llm)

    # define tools
    query_engine_tools.append(
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{wiki_title}",
                description=(
                    "Useful for questions related to specific aspects of"
                    f" {wiki_title} (e.g. the history, arts and culture,"
                    " sports, demographics, or more)."
                ),
            ),
        )
    )

from llama_index.core.agent import AgentRunner
from llama_index.agent.openai import OpenAIAgentWorker, OpenAIAgent
from llama_index.agent.openai import OpenAIAgentWorker

openai_step_engine = OpenAIAgentWorker.from_tools(
    query_engine_tools, llm=llm, verbose=True
)
agent = AgentRunner(openai_step_engine)

response = agent.chat(
    "Summarize the files about Boston.txt and Houston.txt",
)

print(response)