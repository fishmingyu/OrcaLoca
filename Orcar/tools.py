from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import AgentRunner
import subprocess
from typing import List, Union
import datetime
import time

from .executor import LLMCompilerAgentWorker
from .environment.utils import run_bash_in_ctr, get_ctr_from_name
from .environment.utils import ContainerBash

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
    #print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}: Started a bash process")
    #start = time.time()
    process = subprocess.Popen(
        shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output, _ = process.communicate()
    exit_code = process.returncode
    #print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}: Finished a bash process after {time.time()-start} s")
    return f"Exit code: {exit_code}, Output:\n{output.decode()}"

shell_tool = FunctionTool.from_defaults(fn=shell)

def create_tool_list(ctr_bash: Union[ContainerBash, None]) -> List[FunctionTool]:
    tools = [multiply_tool, add_tool]
    if (ctr_bash is None):
        tools.append(shell_tool)
    else:
        def docker_shell(shell_command: str) -> str:
            output = run_bash_in_ctr(ctr_bash, shell_command)
            return output
        docker_shell.__doc__ = shell.__doc__ # Map with same function prompt
        docker_shell_tool = FunctionTool.from_defaults(fn=docker_shell)
        tools.append(docker_shell_tool)
    return tools


def get_llm_compiler_executer(llm: OpenAI, tools: List[FunctionTool]) -> FunctionTool:
    
    callback_manager = llm.callback_manager
    executor = LLMCompilerAgentWorker.from_tools(tools, llm=llm, verbose=True, callback_manager=callback_manager)

    # wrap llm compiler as a tool
    def llm_compiler_tool(input_text: str) -> str:
        """
        This is a tool that wraps the LLM compiler.
        It will analyze the input prompt and plan the actions to take.
        Itself will execute the actions and return the result.
        """
        agent = AgentRunner(executor, callback_manager=callback_manager)
        return agent.chat(input_text)

    return FunctionTool.from_defaults(fn=llm_compiler_tool)
