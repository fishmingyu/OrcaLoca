# ReAct agent formatter

import json
from abc import abstractmethod
from typing import List, Optional, Sequence, Tuple

import tiktoken
from anthropic.types import Usage
from llama_index.core.agent.types import Task, TaskStep
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms.llm import LLM
from llama_index.core.tools import BaseTool
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI

from .log_utils import get_logger
from .prompts import (
    BUG_OUTPUT,
    EXTRACT_EXAMPLES,
    EXTRACT_FIELDS,
    EXTRACT_FORMATS,
    EXTRACT_PROMPTS,
    SEARCH_SYSTEM_HEADER,
    STEP_EXAMPLE,
)
from .types import BaseReasoningStep, SearchActionStep, SearchResult

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

    def __add__(self, other: "TokenCount"):
        return TokenCount(
            in_token_cnt=self.in_token_cnt + other.in_token_cnt,
            out_token_cnt=self.out_token_cnt + other.out_token_cnt,
        )


class TokenCountCached(TokenCount, frozen=True):
    cache_write_cnt: int = 0
    cache_read_cnt: int = 0

    def __add__(self, other: "TokenCountCached"):
        return TokenCountCached(
            in_token_cnt=self.in_token_cnt + other.in_token_cnt,
            out_token_cnt=self.out_token_cnt + other.out_token_cnt,
            cache_read_cnt=self.cache_read_cnt + other.cache_read_cnt,
            cache_write_cnt=self.cache_write_cnt + other.cache_write_cnt,
        )


class TokenCounter:
    """Token counter based on tiktoken / Anthropic"""

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
        if self.encoding is None:
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


class TokenCounterCached(TokenCounter):
    """Token counter with cache based on Anthropic"""

    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)
        assert isinstance(llm, Anthropic)
        self.write_cost_ratio: float = 1.25
        self.read_cost_ratio: float = 0.1
        # TODO: Manually set price data, add function to compute whether cache is worthy

    def equivalent_cost(self, token_count_cached: TokenCountCached) -> TokenCount:
        equi_cost = round(
            token_count_cached.in_token_cnt
            + token_count_cached.cache_write_cnt * self.write_cost_ratio
            + token_count_cached.cache_read_cnt * self.read_cost_ratio
        )
        return TokenCount(
            in_token_cnt=equi_cost,
            out_token_cnt=token_count_cached.out_token_cnt,
        )

    @classmethod
    def is_cache_enabled(cls, llm: LLM) -> bool:
        return isinstance(llm, Anthropic)

    def count_chat(
        self, messages: List[ChatMessage], llm: LLM
    ) -> Tuple[ChatResponse, TokenCountCached]:
        response = llm.chat(
            messages, extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        usage = response.raw["usage"]
        assert isinstance(usage, Usage), f"Unknown usage type: {type(usage)}"
        return (
            response,
            TokenCountCached(
                in_token_cnt=usage.input_tokens,
                out_token_cnt=usage.output_tokens,
                cache_write_cnt=(
                    usage.cache_creation_input_tokens
                    if hasattr(usage, "cache_creation_input_tokens")
                    else 0
                ),
                cache_read_cnt=(
                    usage.cache_read_input_tokens
                    if hasattr(usage, "cache_read_input_tokens")
                    else 0
                ),
            ),
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
        step_type: str,
        tools: Sequence[BaseTool],
        chat_history: List[ChatMessage],
        current_search: Optional[List[SearchResult]] = None,
        current_queue: Optional[List[SearchActionStep]] = None,
    ) -> List[ChatMessage]:
        """Format chat history into list of ChatMessage."""
        assert step_type in [
            "FIRST",
            "REGULAR",
            "CONCLUSION",
        ], f"format: Unknown step type {step_type}"
        is_first = step_type == "FIRST"
        # convert FIRST step to REGULAR step
        if is_first:
            step_type = "REGULAR"
        current_search = current_search or []
        format_args = {
            "tool_desc": "\n".join(get_tool_descriptions(tools)),
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

        # format queue
        # convert queue to list of messages
        status_string = f"""
            Search Queue Status:
            - Queue Length: {len(current_queue)}\n
        """
        for queue_step in current_queue:
            # Search Queue status
            status_string += f"""
                - {queue_step.get_content()}\n
            """
        # logger.info(f"Formatted queue: {status_string}")
        queue_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=status_string,
        )

        output_format: str = "".join(
            json.dumps(
                BUG_OUTPUT if step_type == "CONCLUSION" else STEP_EXAMPLE, indent=4
            )
        )
        fmt_control_msg = ChatMessage(
            role=MessageRole.USER,
            content=(
                f"Please generate next {step_type} step STRICTLY following given format:"
                "<output_format>"
                f"{output_format}"
                "</output_format>"
                "DO NOT SPEAK any REDUNDANT words (like 'json', 'output', etc.)) or thoughts"
            ),
        )

        if is_first:
            return [
                ChatMessage(role=MessageRole.SYSTEM, content=fmt_sys_header),
                *chat_history,
                fmt_control_msg,
            ]
        elif step_type == "REGULAR":
            return [
                ChatMessage(role=MessageRole.SYSTEM, content=fmt_sys_header),
                *chat_history,
                *searching_history,
                queue_message,
                fmt_control_msg,
            ]
        else:
            return [
                ChatMessage(role=MessageRole.SYSTEM, content=fmt_sys_header),
                *chat_history,
                *searching_history,
                fmt_control_msg,
            ]


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
