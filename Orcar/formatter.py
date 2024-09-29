# ReAct agent formatter

import json
import logging
from abc import abstractmethod
from typing import List, Optional, Sequence, Tuple

import tiktoken
from llama_index.core.agent.react.prompts import (
    CONTEXT_REACT_CHAT_SYSTEM_HEADER,
    REACT_CHAT_SYSTEM_HEADER,
)
from llama_index.core.agent.types import Task, TaskStep
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms.llm import LLM
from llama_index.core.tools import BaseTool
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI

from .environment.utils import get_logger
from .prompts import (
    BUG_OUTPUT,
    EXTRACT_EXAMPLES,
    EXTRACT_FIELDS,
    EXTRACT_FORMATS,
    EXTRACT_PROMPTS,
    SEARCH_SYSTEM_HEADER,
    STEP_EXAMPLE,
)
from .search.search_tool import SearchManager
from .types import BaseReasoningStep, ObservationReasoningStep, SearchResult

logger = get_logger(__name__)


def get_react_tool_descriptions(tools: Sequence[BaseTool]) -> List[str]:
    """Tool."""
    tool_descs = []
    for tool in tools:
        tool_desc = (
            f"> Tool Name: {tool.metadata.name}\n"
            f"Tool Description: {tool.metadata.description}\n"
            f"Tool Args: {tool.metadata.fn_schema_str}\n"
        )
        tool_descs.append(tool_desc)
    return tool_descs


class TokenCount(BaseModel, frozen=True):
    """Token count of an LLM call"""

    in_token_cnt: int
    out_token_cnt: int


class TokenCounter:
    """Token counter based on tiktoken"""

    def __init__(self, llm: LLM) -> None:
        model = llm.metadata.model_name
        if isinstance(llm, OpenAI):
            self.encoding = tiktoken.encoding_for_model(model)
        elif isinstance(llm, Anthropic):
            self.encoding = llm.tokenizer
        else:
            raise Exception(f"gen_config: No tokenizer for model {model}")
        logger.info(f"Found tokenizer for model '{model}'")

    def count(self, string: str) -> int:
        if self.encoding == None:
            return 0
        return len(self.encoding.encode(string))

    def count_chat(
        self, messages: List[ChatMessage], llm: LLM
    ) -> Tuple[ChatResponse, TokenCount]:
        in_token_cnt = self.count(llm.messages_to_prompt(messages))
        response = llm.chat(messages)
        out_token_cnt = self.count(response.message.content)
        return (
            response,
            TokenCount(in_token_cnt=in_token_cnt, out_token_cnt=out_token_cnt),
        )


def replace_unicode_quotations(input: str) -> str:
    """
    Claude 3.5 Sonnet sometimes fail to disguish curly quote mark with normal one,
    which can result in json schema fail;
    So we replace all curly quote marks in input.
    """
    unicode_quotations = {
        "“": "\\u201C",
        "”": "\\u201D",
        "‘": "\\u2018",
        "’": "\\u2019",
    }
    for key, value in unicode_quotations.items():
        input = input.replace(key, value)
    return input


# TODO: come up with better name
class BaseAgentChatFormatter(BaseModel):
    """Base chat formatter."""

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def format(
        self,
        tools: Sequence[BaseTool],
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> List[ChatMessage]:
        """Format chat history into list of ChatMessage."""


class ReActChatFormatter(BaseAgentChatFormatter):
    """ReAct chat formatter."""

    system_header: str = REACT_CHAT_SYSTEM_HEADER  # default
    context: str = ""  # not needed w/ default

    def format(
        self,
        tools: Sequence[BaseTool],
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> List[ChatMessage]:
        """Format chat history into list of ChatMessage."""
        current_reasoning = current_reasoning or []

        format_args = {
            "tool_desc": "\n".join(get_react_tool_descriptions(tools)),
            "tool_names": ", ".join([tool.metadata.get_name() for tool in tools]),
        }
        if self.context:
            format_args["context"] = self.context

        fmt_sys_header = self.system_header.format(**format_args)

        # format reasoning history as alternating user and assistant messages
        # where the assistant messages are thoughts and actions and the user
        # messages are observations
        reasoning_history = []
        for reasoning_step in current_reasoning:
            if isinstance(reasoning_step, ObservationReasoningStep):
                message = ChatMessage(
                    role=MessageRole.USER,
                    content=reasoning_step.get_content(),
                )
            else:
                message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=reasoning_step.get_content(),
                )
            reasoning_history.append(message)

        return [
            ChatMessage(role=MessageRole.SYSTEM, content=fmt_sys_header),
            *chat_history,
            *reasoning_history,
        ]

    @classmethod
    def from_defaults(
        cls,
        system_header: Optional[str] = None,
        context: Optional[str] = None,
    ) -> "ReActChatFormatter":
        """Create ReActChatFormatter from defaults."""
        if not system_header:
            system_header = (
                REACT_CHAT_SYSTEM_HEADER
                if not context
                else CONTEXT_REACT_CHAT_SYSTEM_HEADER
            )

        return ReActChatFormatter(
            system_header=system_header,
            context=context or "",
        )

    @classmethod
    def from_context(cls, context: str) -> "ReActChatFormatter":
        """Create ReActChatFormatter from context.

        NOTE: deprecated

        """
        logger.warning(
            "ReActChatFormatter.from_context is deprecated, please use `from_defaults` instead."
        )
        return ReActChatFormatter.from_defaults(
            system_header=CONTEXT_REACT_CHAT_SYSTEM_HEADER, context=context
        )


def get_tool_descriptions(tools: Sequence[BaseTool]) -> List[str]:
    """Tool."""
    tool_descs = []
    for tool in tools:
        tool_desc = (
            f"> Tool Name: {tool.metadata.name}\n"
            f"Tool Description: {tool.metadata.description}\n"
            f"Tool Args: {tool.metadata.fn_schema_str}\n"
        )
        tool_descs.append(tool_desc)
    return tool_descs


class SearchChatFormatter(BaseAgentChatFormatter):
    """ReAct chat formatter."""

    system_header: str = SEARCH_SYSTEM_HEADER  # default

    def format(
        self,
        tools: Sequence[BaseTool],
        chat_history: List[ChatMessage],
        current_search: Optional[List[SearchResult]] = None,
    ) -> List[ChatMessage]:
        """Format chat history into list of ChatMessage."""
        current_search = current_search or []
        format_args = {
            "tool_desc": "\n".join(get_tool_descriptions(tools)),
            "priority_desc": "\n".join(str(SearchManager.search_tool_priority)),
            "step_format": "".join(json.dumps(STEP_EXAMPLE, indent=4)),
            "bug_locations": "".join(json.dumps(BUG_OUTPUT, indent=4)),
        }

        fmt_sys_header = self.system_header.format(**format_args)
        # logger.info(f"Formatted system header: {fmt_sys_header}")

        # format searching history
        searching_history = []
        for searching_step in current_search:
            message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=searching_step.get_content(),
            )
            searching_history.append(message)

        return [
            ChatMessage(role=MessageRole.SYSTEM, content=fmt_sys_header),
            *chat_history,
            *searching_history,
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "Please generate next step STRICTLY following given format, "
                    "DO NOT SPEAK any REDUNDANT words (like 'json', 'output', etc.)) or thoughts"
                ),
            ),
        ]

    @classmethod
    def from_defaults(
        cls,
        system_header: Optional[str] = None,
    ) -> "SearchChatFormatter":
        """Create SearchChatFormatter from defaults."""
        if not system_header:
            system_header = REACT_CHAT_SYSTEM_HEADER

        return SearchChatFormatter(
            system_header=system_header,
        )


class ExtractChatFormatter(BaseAgentChatFormatter):
    """Extractor Agent formatter."""

    def format(self, step: TaskStep, task: Task, handler: str) -> List[ChatMessage]:
        """Format chat history into list of ChatMessage."""
        sysheader = ChatMessage(
            role=MessageRole.SYSTEM, content=EXTRACT_PROMPTS["header"]
        )
        if handler == "slice":
            example = EXTRACT_PROMPTS["example"]
            example_format_args = {
                "example_repo_name": EXTRACT_EXAMPLES[handler]["repo_name"],
                "example_input_description": EXTRACT_EXAMPLES[handler][
                    "input_description"
                ],
                "example_output": "".join(
                    json.dumps(EXTRACT_EXAMPLES[handler]["example_output"], indent=4)
                ),
            }
            fmt_example = example.format(**example_format_args)
            user_msg = EXTRACT_PROMPTS[handler]
            format_args = {
                "output_format": "".join(
                    json.dumps(EXTRACT_FORMATS[handler], indent=4)
                ),
                "output_fields": EXTRACT_FIELDS[handler],
                "example": fmt_example,
                "repo_name": task.extra_state["inst"]["repo"],
                "input_description": replace_unicode_quotations(
                    task.extra_state["inst"]["problem_statement"]
                ),
            }
            fmt_user_msg = user_msg.format(**format_args)
            return [
                sysheader,
                ChatMessage(role=MessageRole.USER, content=fmt_user_msg),
            ]
        elif handler == "parse":
            step_name = step.step_state["name"]
            parse_type = task.extra_state["parse_type"][step_name]
            example = EXTRACT_PROMPTS["example"]
            example_format_args = {
                "example_repo_name": EXTRACT_EXAMPLES[handler][parse_type]["repo_name"],
                "example_input_description": EXTRACT_EXAMPLES[handler][parse_type][
                    "input_description"
                ],
                "example_output": "".join(
                    json.dumps(
                        EXTRACT_EXAMPLES[handler][parse_type]["example_output"],
                        indent=4,
                    )
                ),
            }
            fmt_example = example.format(**example_format_args)
            user_msg = EXTRACT_PROMPTS[handler]
            format_args = {
                "output_format": "".join(
                    json.dumps(EXTRACT_FORMATS[handler], indent=4)
                ),
                "output_fields": EXTRACT_FIELDS[handler],
                "example": fmt_example,
                "repo_name": task.extra_state["inst"]["repo"],
                "input_description": task.extra_state["slices"][step_name],
            }
            fmt_user_msg = user_msg.format(**format_args)
            return [
                sysheader,
                ChatMessage(role=MessageRole.USER, content=fmt_user_msg),
            ]
        elif handler == "judge":
            user_msg = EXTRACT_PROMPTS[handler]
            format_args = {
                "output_format": "".join(
                    json.dumps(EXTRACT_FORMATS[handler], indent=4)
                ),
                "output_fields": EXTRACT_FIELDS[handler],
                "repo_name": task.extra_state["inst"]["repo"],
                "input_description": task.extra_state["inst"]["problem_statement"],
                "reproduce_snippet": task.extra_state["slices"]["reproduce_code_parse"],
                "reproducer_log": task.extra_state["slices"]["reproduce_log_parse"],
            }
            fmt_user_msg = user_msg.format(**format_args)
            return [
                sysheader,
                ChatMessage(role=MessageRole.USER, content=fmt_user_msg),
            ]
        elif handler == "summarize":
            user_msg = EXTRACT_PROMPTS[handler]
            example = EXTRACT_PROMPTS["example"]
            example_format_args = {
                "example_repo_name": EXTRACT_EXAMPLES[handler]["repo_name"],
                "example_input_description": EXTRACT_EXAMPLES[handler][
                    "input_description"
                ],
                "example_output": "".join(
                    json.dumps(
                        EXTRACT_EXAMPLES[handler]["example_output"],
                        indent=4,
                    )
                ),
            }
            fmt_example = example.format(**example_format_args)
            format_args = {
                "output_format": "".join(
                    json.dumps(EXTRACT_FORMATS[handler], indent=4)
                ),
                "output_fields": EXTRACT_FIELDS[handler],
                "example": fmt_example,
                "repo_name": task.extra_state["inst"]["repo"],
                "input_description": task.extra_state["inst"]["problem_statement"],
            }
            fmt_user_msg = user_msg.format(**format_args)
            return [
                sysheader,
                ChatMessage(role=MessageRole.USER, content=fmt_user_msg),
            ]
        raise NotImplementedError
