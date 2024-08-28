# ReAct agent formatter

import logging
import json
from abc import abstractmethod
from typing import List, Optional, Sequence

from llama_index.core.agent.react.prompts import (
    CONTEXT_REACT_CHAT_SYSTEM_HEADER,
    REACT_CHAT_SYSTEM_HEADER,
)
from .prompts import SEARCH_SYSTEM_HEADER
from .prompts import SEARCH_STEP_ANSWER, SEARCH_STEP_EXAMPLE, SEARCH_RESULT, OBSERVATION
from .types import (
    BaseReasoningStep,
    ObservationReasoningStep,
)
from .types import SearchActionStep
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.tools import BaseTool


logger = logging.getLogger(__name__)


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
        current_search: Optional[List[SearchActionStep]] = None,
    ) -> List[ChatMessage]:
        """Format chat history into list of ChatMessage."""
        current_search = current_search or []
        format_args = {
            "tool_desc": "\n".join(get_tool_descriptions(tools)),
            "search_format": "".join(json.dumps(SEARCH_STEP_ANSWER, indent=4)),
            "example_output": "".join(json.dumps(SEARCH_STEP_EXAMPLE, indent=4)),
            "observation": "".join(json.dumps(OBSERVATION, indent=4)),
            "bug_locations": "".join(json.dumps(SEARCH_RESULT, indent=4)),
        }

        fmt_sys_header = self.system_header.format(**format_args)
        print(fmt_sys_header)
        logger.info(f"Formatted system header: {fmt_sys_header}")

        # format searching history
        searching_history = []
        for searching_step in current_search:
            message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=searching_step,
            )
            searching_history.append(message)

        return [
            ChatMessage(role=MessageRole.SYSTEM, content=fmt_sys_header),
            *chat_history,
            *searching_history,
        ]

    @classmethod
    def from_defaults(
        cls,
        system_header: Optional[str] = None,
    ) -> "SearchChatFormatter":
        """Create SearchChatFormatter from defaults."""
        if not system_header:
            system_header = (
                REACT_CHAT_SYSTEM_HEADER
            )

        return SearchChatFormatter(
            system_header=system_header,
        )