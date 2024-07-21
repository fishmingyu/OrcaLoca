import json
from typing import Sequence, List

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from Orcar import OrcarAgentWorker
import subprocess


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

def execute(shell_command: str) -> str:
    """Execute a shell command and return"""
    process = subprocess.Popen(
        shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output, _ = process.communicate()
    exit_code = process.returncode
    return f"Exit code: {exit_code}, Output:\n{output.decode()}"

shell_tool = FunctionTool.from_defaults(fn=execute)


tools = [multiply_tool, add_tool, shell_tool]

from llama_index.core.agent import AgentRunner

llm = OpenAI(model="gpt-4")


callback_manager = llm.callback_manager

# tool bindings
agent_worker = OrcarAgentWorker.from_tools(
    tools, llm=llm, verbose=True, callback_manager=callback_manager
)
agent = AgentRunner(agent_worker, callback_manager=callback_manager)

response = agent.chat("What is (121 * 3) + 42? Calculate step by step. Make a new directory named tests, and then write the calculated result into a new file named 'result.txt' in this directory.")

print(response)