"""Base types for ReAct agent."""

from abc import abstractmethod
from typing import Dict, List
from collections import namedtuple
from llama_index.core.bridge.pydantic import BaseModel


class BaseReasoningStep(BaseModel):
    """Reasoning step."""

    @abstractmethod
    def get_content(self) -> str:
        """Get content."""

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""


class ActionReasoningStep(BaseReasoningStep):
    """Action Reasoning step."""

    thought: str
    action: str
    action_input: Dict

    def get_content(self) -> str:
        """Get content."""
        return (
            f"Thought: {self.thought}\nAction: {self.action}\n"
            f"Action Input: {self.action_input}"
        )

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class ObservationReasoningStep(BaseReasoningStep):
    """Observation reasoning step."""

    observation: str
    return_direct: bool = False

    def get_content(self) -> str:
        """Get content."""
        return f"Observation: {self.observation}"

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return self.return_direct


class ResponseReasoningStep(BaseReasoningStep):
    """Response reasoning step."""

    thought: str
    response: str
    is_streaming: bool = False

    def get_content(self) -> str:
        """Get content."""
        if self.is_streaming:
            return (
                f"Thought: {self.thought}\n"
                f"Answer (Starts With): {self.response} ..."
            )
        else:
            return f"Thought: {self.thought}\n" f"Answer: {self.response}"

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return True
    

bug_dict = namedtuple('bugdict', ["file", "function", "content"])

class SearchStep(BaseReasoningStep):
    """Search reasoning step."""

    search_method: List[str]
    search_bugs: List[bug_dict]

    def get_content(self) -> str:
        """Get content."""
        # recursively format the search method
        search_method = " -> ".join(self.search_method)
        return f"Search Method: {search_method}\n" f"Search Bugs: {self.search_bugs}"

    @property
    def is_done(self) -> bool:
        """
        If the search method is empty, the is_done is False. (no matchings)
        """
        return not self.search_method