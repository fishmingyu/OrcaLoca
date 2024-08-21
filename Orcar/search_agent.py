"""
A search agent. Process raw response into json format.
"""

import json
from typing import Sequence, List, Any

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from openai.types.chat import ChatCompletionMessageToolCall

import logging
from .environment.utils import get_logger
from .search import SearchManager

logger = get_logger("search_agent")

SYSTEM_PROMPT = r"""
You are a helpful assistant that use API calls to report bug code snippets from a text into json format.
You need to extract where are the bug locations by analyzing the text.
There are some API calls that you can use to extract the information.
The API calls include:
def search_func(self, func_name: str) -> Tuple[str, str]:
    \"""Search the function in the knowledge graph.

    Args:
        func_name (str): The function name to search.

    Returns:
        Tuple[str, str]: The file path and the code snippet of the function definition.
    \"""

Provide your answer in JSON structure like this, you should ignore the argument placeholders in api calls.
For example, search_func(func_name="str") should be search_func("str")
Make sure each API call is written as a valid python expression.

{
    "API_calls": ["api_call_1(args)", "api_call_2(args)", ...],
    "bug_locations":[{"file": "path/to/file", "function": "function_name", "content" : "code_snippet"}, {"file": "path/to/file", "function": "function_name", "content" : "code_snippet"} ... ]
}

"""

class SearchAgent:
    def __init__(
        self,
        repo_path: str = None,
        llm: str = "gpt-4o",
        chat_history: List[ChatMessage] = [],
    ) -> None:
        self._repo_path = repo_path
        self._tools : Sequence[BaseTool] = [],
        self._chat_history = chat_history
        self._setup_llm(llm)
        self._setup_search_manager()
        self._setup_tools()

    def _setup_llm(self, llm: str):
        system_prompt = SYSTEM_PROMPT
        self._llm = OpenAI(model=llm, system_prompt=system_prompt)

    def _setup_search_manager(self):
        if self._repo_path is not None:
            self._search_manager = SearchManager(repo_path=self._repo_path)
        else:
            logger.debug("Repo path is not set.")
            raise ValueError("Repo path is required.")
        
    def _setup_tools(self):
        tools = []
        search_tool = FunctionTool.from_defaults(fn=self._search_manager.search_func)
        tools.append(search_tool)
        self._tools = {tool.metadata.name: tool for tool in tools}

    def reset(self) -> None:
        self._chat_history = []

    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))
        tools = [
            tool.metadata.to_openai_tool() for _, tool in self._tools.items()
        ]

        ai_message = self._llm.chat(chat_history, tools=tools).message
        additional_kwargs = ai_message.additional_kwargs
        chat_history.append(ai_message)

        tool_calls = additional_kwargs.get("tool_calls", None)
        # parallel function calling is now supported
        if tool_calls is not None:
            for tool_call in tool_calls:
                function_message = self._call_function(tool_call)
                chat_history.append(function_message)
                ai_message = self._llm.chat(chat_history).message
                chat_history.append(ai_message)

        return ai_message.content

    def _call_function(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> ChatMessage:
        id_ = tool_call.id
        function_call = tool_call.function
        tool = self._tools[function_call.name]
        output = tool(**json.loads(function_call.arguments))
        return ChatMessage(
            name=function_call.name,
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": id_,
                "name": function_call.name,
            },
        )