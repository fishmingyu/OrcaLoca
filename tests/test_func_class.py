from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool

class MathOperations:
    def add(self, a: int, b: int) -> int:
        """Add two integers and return the result integer"""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two integers and return the result integer"""
        return a * b

# Create an instance of the class
math_ops = MathOperations()

# Create tools for the methods
add_tool = FunctionTool.from_defaults(fn=math_ops.add)
multiply_tool = FunctionTool.from_defaults(fn=math_ops.multiply)

# Initialize the LLM (Language Model)
llm = OpenAI(model="gpt-3.5-turbo-instruct")

# Create the ReActAgent with the tools
agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

# Example usage
response = agent.chat("Please add 2 and 3, then multiply the result by 4.")

print(response)
