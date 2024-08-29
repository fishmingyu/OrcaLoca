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
    

class SearchActionStep(BaseReasoningStep):
    """Search action reasoning step."""

    action: str
    action_input: Dict

    def get_content(self) -> str:
        """Get content."""
        return (
            f"Search Action: {self.action}\n"
            f"Search Action Input: {self.action_input}"
        )
    
    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class SearchResult(BaseReasoningStep):
    """Search result reasoning step."""

    search_action: str
    search_input: Dict
    search_content: str

    def get_content(self) -> str:
        """Get content."""
        return (
            f"Search Result: {self.search_action}\n"
            f"Arg Input: {self.search_input}\n"
            f"Search Content: {self.search_content}"
        )

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return True

class SearchObservationStep(BaseReasoningStep):
    """Search observation reasoning step."""

    observation: str
    search_next: List[str] = [],
    is_enough_context: bool = False

    def get_content(self) -> str:
        """Get content."""
        return f"Search Observation: {self.observation}, What to search next: {self.search_next}"

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return self.is_enough_context