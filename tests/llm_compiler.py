import json
from typing import List, Sequence

from llama_index.agent.llm_compiler import LLMCompilerAgentWorker
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.llms.openai import OpenAI


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

tools = [multiply_tool, add_tool]

from llama_index.core.agent import AgentRunner

llm = OpenAI(model="gpt-4o")


callback_manager = llm.callback_manager


agent_worker = LLMCompilerAgentWorker.from_tools(
    tools, llm=llm, verbose=True, callback_manager=callback_manager
)
agent = AgentRunner(agent_worker, callback_manager=callback_manager)

response = agent.chat("What is (121 * 3) + 42?")

print(response)
