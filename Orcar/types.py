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

    search_action: str
    search_action_input: Dict

    def get_content(self) -> str:
        """Get content."""
        return (
            f"Search Action: {self.search_action}\n"
            f"Search Action Input: {self.search_action_input}"
        )

    def __eq__(self, other):
        return (self.search_action == other.search_action) and (
            self.search_action_input == other.search_action_input
        )

    def get_search_input(self) -> str:
        """Get query."""
        """Different search_action
            self.exact_search,
            self.fuzzy_search,
            self.search_file_skeleton,
            self.search_source_code,
        """
        search_input = ""
        if self.search_action == "exact_search":
            # check if containing_class is in search_action_input (Dict)
            if "containing_class" in self.search_action_input:
                # avoid query==containing_class
                if (
                    self.search_action_input["query"]
                    == self.search_action_input["containing_class"]
                ):
                    search_input = f"{self.search_action_input['file_path']}::{self.search_action_input['query']}"
                else:
                    search_input = f"{self.search_action_input['file_path']}::{self.search_action_input['containing_class']}::{self.search_action_input['query']}"
            else:
                search_input = f"{self.search_action_input['file_path']}::{self.search_action_input['query']}"
        elif self.search_action == "fuzzy_search":
            search_input = self.search_action_input["query"]
        elif self.search_action == "search_file_skeleton":
            search_input = self.search_action_input["file_name"]
        elif self.search_action == "search_source_code":
            search_input = self.search_action_input["query"]
        return search_input


class EditActionStep(BaseReasoningStep):
    """Edit action reasoning step."""

    action_input: Dict

    def get_content(self) -> str:
        """Get content."""
        return f"Edit Action Input: {self.action_input}"


class SearchResult(SearchActionStep):
    """Search result reasoning step."""

    search_content: str

    def get_content(self) -> str:
        """Get content."""
        return f"""<Search Result>\n
            Search Action: {self.search_action}\n
            Search Action Input: {self.search_action_input}\n
            {self.search_content}\n</Search Result>"""

    def __eq__(self, other):  # not used
        pass


class HeuristicSearchResult(BaseModel):
    """Heuristic search result reasoning step."""

    heuristic: float
    search_result: SearchResult

    def get_content(self) -> str:
        """Get content."""
        # cut off the first 50 characters of the search content
        search_content = self.search_result.get_content()
        return f"Heuristic: {self.heuristic}\n" f"{search_content[:50]}"

    def __lt__(self, other):
        return self.heuristic < other.heuristic


class BugLocations(BaseModel):
    """Bug locations reasoning step."""

    file_name: str
    class_name: str
    method_name: str

    def bug_query(self) -> str | None:
        """Get bug query."""
        # class_name can be "", method_name can also be ""
        if self.file_name == "":
            return None
        if self.class_name != "" and self.method_name != "":
            return f"{self.file_name}::{self.class_name}::{self.method_name}"
        elif self.class_name == "" and self.method_name != "":
            return f"{self.file_name}::{self.method_name}"
        elif self.class_name != "" and self.method_name == "":
            return f"{self.file_name}::{self.class_name}"
        else:
            return f"{self.file_name}"


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


class CodeInfoWithClass(CodeInfo):
    """Code keyword and location info with class"""

    class_name: str


class ExtractParseStep(BaseReasoningStep):
    """Extract parse step"""

    code_info_list: List[CodeInfo]

    def get_content(self) -> str:
        """Get content."""
        return f"code_info_list: {self.code_info_list}\n"


class ExtractJudgeStep(BaseReasoningStep):
    """Extract summarize step"""

    is_successful: bool
    fixed_reproduce_snippet: str

    def get_content(self) -> str:
        """Get content."""
        return (
            f"is_successful: {self.is_successful}\n"
            f"fixed_reproduce_snippet: {self.fixed_reproduce_snippet}\n"
        )


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
    suspicious_code: List[CodeInfo] = []
    suspicious_code_from_tracer: List[CodeInfoWithClass] = []
    related_source_code: str = ""
    is_reproduce_pass: bool = False
    reproduce_code: str = ""
    env_reproduce_path: str = ""


class SearchInput(BaseModel):
    """
    Search input
    """

    problem_statement: str
    extract_output: ExtractOutput

    def get_content(self) -> str:
        """Get content."""
        suspicious_code = ", ".join(
            f"{code.keyword}" for code in self.extract_output.suspicious_code
        )
        suspicious_code_from_tracer = ", ".join(
            f"{code.keyword}"
            for code in self.extract_output.suspicious_code_from_tracer
        )
        summary = self.extract_output.summary

        if len(self.extract_output.suspicious_code_from_tracer) == 0:
            return (
                f"Problem Statement: {self.problem_statement}\n"
                f"Suspicious Keyword: {suspicious_code}\n"
                f"Summary: {summary}\n"
            )
        else:
            return (
                f"Problem Statement: {self.problem_statement}\n"
                f"Suspicious Keyword: {suspicious_code}\n"
                f"""Suspicious Keyword from Tracer:
                The following func/class names are more likely to be related to the bug, since they are called by reproducer code.
                We had already put them in the search queue for you.
                {suspicious_code_from_tracer} \n"""
                f"Summary: {summary}\n"
            )


class SearchOutput(BaseModel):
    """
    search_agent output
    """

    bug_locations: List[BugLocations] = []


class EditInput(BaseModel):
    """
    Edit input
    """

    problem_statement: str
    bug_locations: List[BugLocations]


class EditOutput(BaseModel):
    """The output of the Edit prompt."""

    feedback: str
    action_input: Dict

    def get_content(self) -> str:
        """Get content."""
        return f"Feedback: {self.feedback}\n" f"Action Input: {self.action_input}"


class VerifyOutput(BaseModel):
    """The output of the Verify prompt."""

    is_error: bool
    error_msg: str
    verify_log: str

    def get_content(self) -> str:
        """Get content."""
        return (
            f"is_error: {self.is_error}\n"
            f"Error Message: {self.error_msg}\n"
            f"Verify Log: {self.verify_log}"
        )
