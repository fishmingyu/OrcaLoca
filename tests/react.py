from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
import subprocess

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
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

llm = OpenAI(model="gpt-4o")
agent = ReActAgent.from_tools(llm=llm, tools=[multiply_tool, add_tool, shell_tool], verbose=True)


json_text = """
I want to set my current time zone to UTC+0. Can you help me?
"""

response = agent.chat(json_text)
print(response)