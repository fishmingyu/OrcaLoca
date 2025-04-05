"""Base types for ReAct agent."""

import heapq
import re
from abc import abstractmethod
from typing import Dict, List, Tuple

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

    def __hash__(self):
        return hash((self.search_action, frozenset(self.search_action_input.items())))

    def get_search_input(self) -> str:
        """Get query."""
        """Different search_action
            self.search_class,
            self.search_method_in_class,
            self.search_callable,
            self.search_file_contents,
            self.search_source_code,
        """
        search_input = ""
        if self.search_action == "search_class":
            class_name = self.search_action_input["class_name"]
            if "file_path" in self.search_action_input:
                search_input = f"{self.search_action_input['file_path']}::{class_name}"
            else:
                search_input = self.search_action_input["class_name"]
        elif self.search_action == "search_method_in_class":
            class_name = self.search_action_input["class_name"]
            method_name = self.search_action_input["method_name"]
            if "file_path" in self.search_action_input:
                search_input = f"{self.search_action_input['file_path']}::{class_name}::{method_name}"
            else:
                search_input = f"{class_name}::{method_name}"
        elif self.search_action == "search_callable":
            query_name = self.search_action_input["query_name"]
            if "file_path" in self.search_action_input:
                search_input = f"{self.search_action_input['file_path']}::{query_name}"
            else:
                search_input = self.search_action_input["query_name"]
        elif self.search_action == "search_file_contents":
            file_name = self.search_action_input["file_name"]
            if "directory_path" in self.search_action_input:
                search_input = (
                    f"{self.search_action_input['directory_path']}/{file_name}"
                )
            else:
                search_input = self.search_action_input["file_name"]
        elif self.search_action == "search_source_code":
            search_input = self.search_action_input["source_code"]
        return search_input


class SearchActionHistory:
    def __init__(self):
        # Initialize an empty dictionary
        self.action_history: Dict[SearchActionStep, int] = {}

    def add_action(self, action: SearchActionStep):
        """
        Add a SearchActionStep to the history.

        If the action matches a key in the dictionary, increment the value.
        If a key matches the new query, increment the value.
        Otherwise, add the action as a new key with an initial value of 1.
        """
        for existing_action in self.action_history.keys():
            # Case 1: Check if the key matches the new query
            if existing_action == action:
                self.action_history[existing_action] += 1
                return

        # Case 2: Add the new query as a key
        self.action_history[action] = 1

    def get_action_count(self, action: SearchActionStep) -> int:
        """Get the count of a specific action."""
        return self.action_history.get(
            action, 0
        )  # return 0 if the action is not in the history

    def keys(self) -> List[SearchActionStep]:
        """Get the keys of the action history."""
        return list(self.action_history.keys())

    def check_action(self, action: SearchActionStep) -> bool:
        """Check if the action is in the history."""
        return action in self.action_history

    def get_history(self):
        """Get the current action history."""
        return self.action_history

    def __repr__(self) -> str:
        """Developer-friendly string representation of the action history."""
        return (
            "SearchActionHistory(\n"
            + "\n".join(
                f"  {action.get_content()}: {count}"
                for action, count in self.action_history.items()
            )
            + "\n)"
        )


# use negative value to ensure that the action with the highest count is popped first
class SearchQueue:
    def __init__(self, priority_config: Dict):
        """Initialize the SearchQueue with a reference to SearchActionHistory."""
        self.queue: List[Tuple[int, int, SearchActionStep]] = (
            []
        )  # (value, order, action)
        self.counter = 0  # To ensure stable sorting for actions with the same value
        self.priority_config = priority_config

    def append(self, action: SearchActionStep):
        """
        Add a new SearchActionStep to the queue with value = 1.
        """
        heapq.heappush(
            self.queue, (-self.priority_config["basic"], self.counter, action)
        )  # (max heap)
        self.counter += 1

    def append_with_priority(self, action: SearchActionStep, priority: float):
        """
        Add a new SearchActionStep to the queue with value priority
        """
        # if enable:
        if self.priority_config["enable"]:
            heapq.heappush(self.queue, (-priority, self.counter, action))
        else:
            heapq.heappush(
                self.queue, (-self.priority_config["basic"], self.counter, action)
            )
        self.counter += 1

    def resort(self, action_history: SearchActionHistory):
        """
        Recheck the count in SearchActionHistory and update the priority queue.
        If the count for a SearchActionStep in SearchActionHistory surpasses the
        value for that step in the queue, update the value to the count.
        """
        # Create a new list for updated queue
        updated_queue = []
        while self.queue:
            neg_value, order, action = heapq.heappop(self.queue)
            current_count = action_history.get_action_count(action)

            # If the count surpasses the current value, update it
            if current_count > -neg_value:
                neg_value = -current_count

            updated_queue.append((neg_value, order, action))

        # Rebuild the heap
        self.queue = updated_queue
        heapq.heapify(self.queue)

    def pop(self) -> SearchActionStep:
        """
        Remove and return the SearchActionStep with the biggest value. (minimum neg_value)
        """
        if not self.queue:
            raise IndexError("pop from an empty priority queue")
        _, _, action = heapq.heappop(self.queue)
        return action

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the queue."""
        return (
            "SearchQueue(\n"
            + "\n".join(
                f"  Value: {-neg_value}, Action: {action.get_content()}"
                for neg_value, _, action in sorted(self.queue)
            )
            + "\n)"
        )

    def __len__(self) -> int:
        """Return the length of the queue."""
        return len(self.queue)

    def __iter__(self):
        """Return an iterator for the queue."""
        return iter(sorted(self.queue))


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

    def get_next_response(self) -> str:
        """Get next response."""
        return f"""<New Info>\n
            Search Action: {self.search_action}\n
            Search Action Input: {self.search_action_input}\n
            {self.search_content}\n</New Info>"""

    def __eq__(self, other):  # not used
        pass


class HeuristicSearchResult(BaseModel):
    """Heuristic search result"""

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
    """Bug locations"""

    file_path: str
    class_name: str
    method_name: str

    def bug_query(self) -> str | None:
        """Get bug query."""
        # class_name can be "", method_name can also be ""
        if self.file_path == "":
            return None
        if self.class_name != "" and self.method_name != "":
            return f"{self.file_path}::{self.class_name}::{self.method_name}"
        elif self.class_name == "" and self.method_name != "":
            return f"{self.file_path}::{self.method_name}"
        elif self.class_name != "" and self.method_name == "":
            return f"{self.file_path}::{self.class_name}"
        else:
            return f"{self.file_path}"


class DependencyLoc(BaseModel):
    """Dependency location"""

    file_path: str
    line_range: str

    def line_range_tuple(self) -> list[int]:
        """Get line range list."""
        # input example, [261, 271]
        # first use regex to extract the line range
        regex = r"\[(\d+), (\d+)\]"
        match = re.search(regex, self.line_range)
        if match:
            return [int(match.group(1)), int(match.group(2))]


class BugLocationsWithLine(BugLocations):
    """Bug locations with line range."""

    line_range: str

    def line_range_tuple(self) -> list[int]:
        """Get line range list."""
        # input example, [261, 271]
        # first use regex to extract the line range
        regex = r"\[(\d+), (\d+)\]"
        match = re.search(regex, self.line_range)
        if match:
            return [int(match.group(1)), int(match.group(2))]


class TraceAnalysisSliceStep(BaseReasoningStep):
    """Trace Analysis slice step"""

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


class TraceAnalysisParseStep(BaseReasoningStep):
    """Trace Analysis parse step"""

    code_info_list: List[CodeInfo]

    def get_content(self) -> str:
        """Get content."""
        return f"code_info_list: {self.code_info_list}\n"


class TraceAnalysisJudgeStep(BaseReasoningStep):
    """Trace Analysis summarize step"""

    is_successful: bool
    fixed_reproduce_snippet: str

    def get_content(self) -> str:
        """Get content."""
        return (
            f"is_successful: {self.is_successful}\n"
            f"fixed_reproduce_snippet: {self.fixed_reproduce_snippet}\n"
        )


class TraceAnalysisSummarizeStep(BaseReasoningStep):
    """Trace Analysis summarize step"""

    summary: str
    code_info_list: List[CodeInfo]

    def get_content(self) -> str:
        """Get content."""
        return f"summary: {self.summary}\n" f"code_info_list: {self.code_info_list}\n"


class TraceAnalysisOutput(BaseModel):
    """
    Trace Analysis agent output
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
    trace_analysis_output: TraceAnalysisOutput

    def get_content(self) -> str:
        """Get content."""
        # suspicious_code = ", ".join(
        #     f"{code.keyword}" for code in self.trace_analysis_output.suspicious_code
        # )
        suspicious_code_from_tracer = ", ".join(
            f"{code.keyword}"
            for code in self.trace_analysis_output.suspicious_code_from_tracer
        )
        # summary = self.trace_analysis_output.summary

        if len(self.trace_analysis_output.suspicious_code_from_tracer) == 0:
            return (
                f"<Problem Statement>: {self.problem_statement}\n </Problem Statement>"
                # f"Suspicious Keyword: {suspicious_code}\n"
                # f"Summary: {summary}\n"
            )
        else:
            return (
                f"<Problem Statement>: {self.problem_statement}\n </Problem Statement>"
                # f"Suspicious Keyword: {suspicious_code}\n"
                f"""Suspicious Keyword from Tracer:
                The following func/class names are more likely to be related to the bug, since they are called by reproducer code.
                We had already put them in the search queue for you.
                {suspicious_code_from_tracer} \n"""
                # f"Summary: {summary}\n"
            )


class SearchOutput(BaseModel):
    """
    search_agent output
    """

    conclusion: str = ""
    bug_locations: List[BugLocations] = []


class SearchOutputTopK(BaseModel):
    """
    search_agent output for top-k retrieval mode

    Contains:
    - conclusion: Summary of findings
    - bug_locations: List of potential bug locations (methods/functions and files)
    - top_files_retrieved: List of unique file paths that were considered relevant
    """

    conclusion: str = ""
    bug_locations: List[BugLocations] = []
    top_files_retrieved: List[str] = []


class EditInput(BaseModel):
    """
    Edit input
    """

    problem_statement: str
    hint: str
    bug_locations: List[BugLocationsWithLine]
    dependency: List[DependencyLoc]


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
