import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Union

from langchain_core.runnables import (
    chain as as_runnable,
)
from typing_extensions import TypedDict
from typing import Sequence

from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

import getpass
import os


def _get_pass(var: str):
    if var not in os.environ:
        os.environ[var] = getpass.getpass(f"{var}: ")


# Optional: Debug + trace calls using LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LLMCompiler"
_get_pass("LANGCHAIN_API_KEY")
_get_pass("OPENAI_API_KEY")

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import subprocess
from Orcar import get_math_tool

calculate = get_math_tool(ChatOpenAI(model="gpt-4-turbo-preview"))
search = DuckDuckGoSearchResults(
    max_results=1,
    description='duckduckgo_results_json(query="the search query") - a search engine.',
)

@tool
def shell_tool(shell_command: str) -> str:
    """Execute a shell command and return"""
    process = subprocess.Popen(
        shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output, _ = process.communicate()
    exit_code = process.returncode

    return f"Exit code: {exit_code}, Output:\n{output.decode()}"



tools = [search, calculate, shell_tool]
llm = ChatOpenAI(model="gpt-4-turbo-preview")

calculate.invoke(
    {
        "problem": "What's the temp of sf + 5?",
        "context": ["Thet empreature of sf is 32 degrees"],
    }
)

from Orcar import OrcarAgent

agent = OrcarAgent(tools)

for step in agent.stream([HumanMessage(content="What is 20+(2*4)? Calculate step by step. Make a new directory named tests, and then write the calculated result into a new file named 'result.txt' in this directory.")]):
    print(step)
    print("---")


# agent.stream([HumanMessage(content="What is the sum of New York's GDP in 2022 and 2023?")])
# agent.invoke([HumanMessage(content="What is 20+(2*4)? Calculate step by step. Make a new directory named tests, and then write the calculated result into a new file named 'result.txt' in this directory." )])