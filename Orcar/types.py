"""Base types for ReAct agent."""

from abc import abstractmethod
from typing import Dict, List

from llama_index.core.bridge.pydantic import BaseModel


class BaseReasoningStep(BaseModel):
    """Reasoning step."""

    @abstractmethod
    def get_content(self) -> str:
        """Get content."""


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


class SearchResult(BaseReasoningStep):
    """Search result reasoning step."""

    search_action: str
    search_action_input: Dict
    search_content: str

    def get_content(self) -> str:
        """Get content."""
        return (
            f"Search Action: {self.search_action}\n"
            f"Search Action Input: {self.search_action_input}\n"
            f"\n {self.search_content}"
        )

    def get_query(self) -> str:
        """Get query."""
        """Different search_action
            self.search_class_skeleton,
            self.search_method_in_class,
            self.search_file_skeleton,
            self.search_callable,
            self.search_source_code,
        """
        query_key = ""
        for action in self.search_action:
            if action == "search_class_skeleton":
                query_key = "class_name"
            elif action == "search_method_in_class":
                query_key = "method"
            elif action == "search_file_skeleton":
                query_key = "file_name"
            elif action == "search_callable":
                query_key = "query"
            elif action == "search_source_code":
                query_key = "file_path"

        return self.search_action_input[query_key]


class BugLocations(BaseModel):
    """Bug locations reasoning step."""

    file_name: str
    class_name: str
    method_name: str


class ExtractSliceStep(BaseReasoningStep):
    """Extract slice step"""

    traceback_warning_log_slice: str
    issue_reproducer_slice: str
    source_code_slice: str

    def get_content(self) -> str:
        """Get content."""
        return (
            f"traceback_warning_log_slice: {self.traceback_warning_log_slice}\n"
            f"issue_reproducer_slice: {self.issue_reproducer_slice}\n"
            f"source_code_slice: {self.source_code_slice}\n"
        )


class CodeInfo(BaseModel, frozen=True):
    """Code keyword and location info"""

    keyword: str
    file_path: str


class ExtractParseStep(BaseReasoningStep):
    """Extract parse step"""

    code_info_list: List[CodeInfo]

    def get_content(self) -> str:
        """Get content."""
        return f"code_info_list: {self.code_info_list}\n"


class ExtractJudgeStep(BaseReasoningStep):
    """Extract summarize step"""

    is_successful: bool

    def get_content(self) -> str:
        """Get content."""
        return f"is_successful: {self.is_successful}\n"


class ExtractSummarizeStep(BaseReasoningStep):
    """Extract summarize step"""

    summary: str
    code_info_list: List[CodeInfo]

    def get_content(self) -> str:
        """Get content."""
        return f"summary: {self.summary}\n" f"code_info_list: {self.code_info_list}\n"


class ExtractOutput(BaseModel):
    """
    Extract agent output
    """

    summary: str = ""
    suspicous_code: List[CodeInfo] = []
    suspicous_code_from_tracer: List[CodeInfo] = []
    related_source_code: str = ""


class SearchInput(BaseModel):
    """
    Search input
    """

    problem_statement: str
    extract_output: ExtractOutput

    def get_content(self) -> str:
        """Get content."""
        suspicous_code = ", ".join(
            f"{code.keyword}" for code in self.extract_output.suspicous_code
        )
        summary = self.extract_output.summary
        return (
            f"Problem Statement: {self.problem_statement}\n"
            f"Suspicious Keyword: {suspicous_code}\n"
            f"Summary: {summary}\n"
        )
