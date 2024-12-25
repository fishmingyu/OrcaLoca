"""LLM Compiler Output Parser."""

import io
import json
import re
import sys
import traceback
from typing import Dict, List, Tuple

from llama_index.core.types import BaseOutputParser

from .log_utils import get_logger
from .types import (
    BaseReasoningStep,
    BugLocations,
    CodeInfo,
    EditOutput,
    ExtractJudgeStep,
    ExtractParseStep,
    ExtractSliceStep,
    ExtractSummarizeStep,
    SearchActionStep,
)

logger = get_logger(__name__)

THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
# $1 or ${1} -> 1
ID_PATTERN = r"\$\{?(\d+)\}?"

END_OF_PLAN = "<END_OF_PLAN>"
JOINER_REPLAN = "Replan"


def default_dependency_rule(idx: int, args: str) -> bool:
    """Default dependency rule."""
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


def extract_tool_use(input_text: str) -> Tuple[str, str, str]:
    pattern = (
        r"\s*Thought: (.*?)\n+Action: ([a-zA-Z0-9_]+).*?\n+Action Input: .*?(\{.*\})"
    )

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract tool use from input text: {input_text}")

    thought = match.group(1).strip()
    action = match.group(2).strip()
    action_input = match.group(3).strip()
    return thought, action, action_input


def action_input_parser(json_str: str) -> dict:
    processed_string = re.sub(r"(?<!\w)\'|\'(?!\w)", '"', json_str)
    pattern = r'"(\w+)":\s*"([^"]*)"'
    matches = re.findall(pattern, processed_string)
    return dict(matches)


def extract_final_response(input_text: str) -> Tuple[str, str]:
    pattern = r"\s*Thought:(.*?)Answer:(.*?)(?:$)"

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(
            f"Could not extract final answer from input text: {input_text}"
        )

    thought = match.group(1).strip()
    answer = match.group(2).strip()
    return thought, answer


def load_with_escape(input_text: str) -> dict:
    # Define the regex to match backslashes followed by 's'
    pattern = r"((?:\\)+)s"

    def replace_match(match):
        # Get the matched backslashes and count them
        backslashes = match.group(1)
        # Check if the total backslashes are odd
        if len(backslashes) % 4 == 1:
            # Increase the count by 1 and append 's'
            return backslashes + r"\\" + r"s"
        elif len(backslashes) % 4 == 3:
            # Decrease the count by 1 and append 's'
            return backslashes[:-1] + r"s"
        # If even, return unchanged
        return match.group(0)

    # Apply the regex with the replacement function
    input_text = re.sub(pattern, replace_match, input_text)

    # Parse JSON
    data = json.loads(input_text)

    return data


class SearchOutputParser(BaseOutputParser):
    """ReAct Output parser."""

    def parse(self, output: str, method: str):
        if method == "explore":
            return self.parse_explore(output)
        elif method == "bug_report":
            return self.parse_bug_report(output)

    def parse_explore(
        self, output: str
    ) -> Tuple[str, List[BugLocations], List[SearchActionStep]]:
        """Parse output from Search agent.

        We expect the output to be the following format:
            "observation": "str",
            "potential_bug_locations": [
                {
                    "file_path": "path/to/file",
                    "class_name": "class_name",
                    "method_name": "function_name",
                },
                {
                    "file_path": "path/to/file",
                    "class_name": "class_name",
                    "method_name": "function_name",
                },
            ],
            "action_lists": [
                {
                    "action": "search_api1",
                    "action_input": {
                    "arg1": "str"
                    }
                },
                {
                    "action": "search_api2",
                    "action_input": {
                    "arg1": "print",
                    "arg2": "add(2, 3)"
                    }
                },
            ]
        """
        if "observation_feedback" in output:
            action_list: List[SearchActionStep] = []
            bug_list: List[BugLocations] = []
            # cast the output to SearchActionStep
            # escape \s in the json string

            json_str = load_with_escape(output)
            observation_json = json_str["observation_feedback"]
            # add <Observation> and </Observation> to the observation
            observation = f"<Observation>\n{observation_json}\n</Observation>"

            for bug_location in json_str["potential_bug_locations"]:
                bug = BugLocations(
                    file_path=bug_location["file_path"],
                    class_name=bug_location["class_name"],
                    method_name=bug_location["method_name"],
                )
                bug_list.append(bug)
            for action in json_str["new_search_actions"]:
                action_list.append(
                    SearchActionStep(
                        search_action=action["action"],
                        search_action_input=action["action_input"],
                    )
                )
            return observation, bug_list, action_list
        else:
            # raise an error if the output is not in the expected format
            raise ValueError(f"Could not parse search action output: {output}")

    def parse_bug_report(self, output: str) -> List[Dict[str, str]]:
        """
        "bug_locations": [
                {
                    "file_path": "path/to/file",
                    "class_name": "class_name",
                    "method_name": "function_name",
                },
                {
                    "file_path": "path/to/file",
                    "class_name": "class_name",
                    "method_name": "function_name",
                },
            ]
        """
        if "bug_locations" in output:
            # cast the output to SearchResult
            search_result = json.loads(output)
            return search_result
        else:
            # raise an error if the output is not in the expected format
            raise ValueError(f"Could not parse bug report output: {output}")


class EditOutputParser(BaseOutputParser):
    """Edit Output parser."""

    def parse(self, output: str) -> EditOutput:
        """Parse output from Edit agent.

        The input is like:
        {
            "feedback": "observation",
            "action_input": {"command": "search_func", "path": "str"},
        }
        """
        if "feedback" in output:
            # cast the output to EditResult
            edit_result = json.loads(output)
            return EditOutput(
                feedback=edit_result["feedback"],
                action_input=edit_result["action_input"],
            )
        else:
            # raise an error if the output is not in the expected format
            raise ValueError(f"Could not parse edit output: {output}")


class ExtractOutputParser(BaseOutputParser):
    """Extractor Agent formatter."""

    def parse(self, output: str, method: str) -> BaseReasoningStep:
        try:
            json_obj: Dict = json.loads(output, strict=False)
        except json.JSONDecodeError:
            with io.StringIO() as err_msg_io:
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info, file=err_msg_io)
                err_msg_io.write("Trying to escape all backslashes")
                logger.info(err_msg_io.getvalue())
            json_obj: Dict = json.loads(output.replace("\\", r"\\"), strict=False)
        if method == "slice":
            return ExtractSliceStep(
                traceback_warning_log_slice=json_obj["traceback_warning_log_slice"],
                issue_reproducer_slice=json_obj["issue_reproducer_slice"],
                source_code_slice=json_obj["source_code_slice"],
            )
        elif method == "parse":
            code_info_list: List[CodeInfo] = [
                CodeInfo(keyword=x["keyword"], file_path=x["file_path"])
                for x in json_obj["code_info_list"]
            ]
            return ExtractParseStep(code_info_list=code_info_list)
        elif method == "judge":
            return ExtractJudgeStep(
                is_successful=json_obj["is_successful"],
                fixed_reproduce_snippet=json_obj["fixed_reproduce_snippet"],
            )
        elif method == "summarize":
            code_info_list: List[CodeInfo] = [
                CodeInfo(keyword=x["keyword"], file_path=x["file_path"])
                for x in json_obj["code_info_list"]
            ]
            return ExtractSummarizeStep(
                summary=json_obj["summary"], code_info_list=code_info_list
            )
        raise NotImplementedError
