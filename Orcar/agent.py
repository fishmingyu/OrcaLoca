from .instructor import ReActAgent
from typing import Any, Dict, List, Sequence, Tuple

from llama_index.core.agent import AgentRunner
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool, FunctionTool
import subprocess

llm = OpenAI(model="gpt-4")

callback_manager = llm.callback_manager

from .executor import LLMCompilerAgentWorker

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b
multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

def shell(shell_command: str) -> str:
    """
    Executes a shell command and returns the output (result).
    """
    process = subprocess.Popen(
        shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output, _ = process.communicate()
    exit_code = process.returncode
    return f"Exit code: {exit_code}, Output:\n{output.decode()}"

shell_tool = FunctionTool.from_defaults(fn=shell)

executor = LLMCompilerAgentWorker.from_tools([multiply_tool, add_tool, shell_tool], llm=llm, verbose=True, callback_manager=callback_manager)

# wrap llm compiler as a tool
def llm_compiler_tool(input_text: str) -> str:
    """
    This is a tool that wraps the LLM compiler.
    It will analyze the input prompt and plan the actions to take.
    Itself will execute the actions and return the result.
    """
    agent = AgentRunner(executor, callback_manager=callback_manager)
    return agent.chat(input_text)

llm_compiler_tool = FunctionTool.from_defaults(fn=llm_compiler_tool)

instructor = ReActAgent.from_tools(llm=llm, tools=[llm_compiler_tool], verbose=True)

# construct the Orcar agent

class OrcarAgent:
    """Orcar agent worker."""
    def __init__(self):
        super().__init__()
        self.llm = llm
        self.instructor = instructor

    def chat(self, text: str) -> str:
        """Chat with the agent."""
        response = self.instructor.chat(text)
        return response

    def run(self, text: str) -> str:
        """Run the agent."""
        response = self.chat(text)
        return response

    def __call__(self, text: str) -> str:
        """Call the agent."""
        return self.run(text)