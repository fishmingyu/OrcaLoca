"""Base types for ReAct agent."""

from abc import abstractmethod
from typing import Dict, List, Tuple
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
    search_next: List[Dict[str, str]] = []  # Updated to a list of dictionaries
    is_enough_context: bool = False

    def get_content(self) -> str:
        """Get content."""
        search_items = ", ".join(f"{key}: {value}" for item in self.search_next for key, value in item.items())
        if not search_items:
            return f"Observation Feedback: {self.observation} Nothing to search next"
        return f"Observation Feedback: {self.observation}, What to search next: {search_items}"

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return self.is_enough_context
    
class ExtractSliceStep(BaseReasoningStep):
    """Extract slice step"""
    traceback_warning_log_slice: str
    issue_reproducer_slice: str
    source_code_slice: str
    natural_language_description_slice: str

    def get_content(self) -> str:
        """Get content."""
        return (
            f"traceback_warning_log_slice: {self.traceback_warning_log_slice}\n"
            f"issue_reproducer_slice: {self.issue_reproducer_slice}\n"
            f"source_code_slice: {self.source_code_slice}\n"
            f"natural_language_description_slice: {self.natural_language_description_slice}"
        )

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False
    
class CodeInfo(BaseModel, frozen=True):
    """Code keyword and location info"""
    keyword: str
    file_path: str

class ExtractParseStep(BaseReasoningStep):
    """Extract parse step"""
    code_info_list: List[CodeInfo]

    def get_content(self) -> str:
        """Get content."""
        return (
            f"code_info_list: {self.code_info_list}\n"
        )

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False
    
class ExtractJudgeStep(BaseReasoningStep):
    """Extract summarize step"""
    is_successful: bool

    def get_content(self) -> str:
        """Get content."""
        return (
            f"is_successful: {self.is_successful}\n"
        )

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False
    
class ExtractSummarizeStep(BaseReasoningStep):
    """Extract summarize step"""
    summary: str

    def get_content(self) -> str:
        """Get content."""
        return (
            f"summary: {self.summary}\n"
        )

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False
    
class ExtractOutput(BaseModel):
    """Extract agent output"""
    summary: str
    suspicous_code: List[CodeInfo]