"""LLM Compiler Output Parser."""

import re
import json

from typing import Any, Dict, List, Sequence, Tuple

from llama_index.core.tools import BaseTool
from llama_index.core.types import BaseOutputParser

from .schema import JoinerOutput, LLMCompilerParseResult
from .utils import get_graph_dict

from .types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ResponseReasoningStep,
    SearchActionStep,
    SearchObservationStep,
    ExtractSliceStep,
    CodeInfo,
    ExtractParseStep,
    ExtractJudgeStep,
    ExtractSummarizeStep,
)
from llama_index.core.output_parsers.utils import extract_json_str
from llama_index.core.types import BaseOutputParser

import logging
from .environment.utils import get_logger

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


class LLMCompilerPlanParser(BaseOutputParser):
    """LLM Compiler plan output parser.

    Directly adapted from source code: https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/output_parser.py.

    """

    def __init__(self, tools: Sequence[BaseTool]):
        """Init params."""
        self.tools = tools

    def parse(self, text: str) -> Dict[int, Any]:
        # 1. search("Ronaldo number of kids") -> 1, "search", '"Ronaldo number of kids"'
        # pattern = r"(\d+)\. (\w+)\(([^)]+)\)"
        pattern = rf"(?:{THOUGHT_PATTERN}\n)?{ACTION_PATTERN}"
        matches = re.findall(pattern, text)

        # convert matches to a list of LLMCompilerParseResult
        results: List[LLMCompilerParseResult] = []
        for match in matches:
            thought, idx, tool_name, args, _ = match
            idx = int(idx)
            results.append(
                LLMCompilerParseResult(
                    thought=thought, idx=idx, tool_name=tool_name, args=args
                )
            )

        # get graph dict
        return get_graph_dict(results, self.tools)


### Helper functions


class LLMCompilerJoinerParser(BaseOutputParser):
    """LLM Compiler output parser for the join step.

    Adapted from _parse_joiner_output in
    https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/llm_compiler.py

    """

    def parse(self, text: str) -> JoinerOutput:
        """Parse."""
        thought, answer, is_replan = "", "", False  # default values
        raw_answers = text.split("\n")
        for answer in raw_answers:
            if answer.startswith("Action:"):
                answer = answer[answer.find("(") + 1 : answer.find(")")]
                is_replan = JOINER_REPLAN in answer
            elif answer.startswith("Thought:"):
                thought = answer.split("Thought:")[1].strip()
        return JoinerOutput(thought=thought, answer=answer, is_replan=is_replan)


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


def parse_action_reasoning_step(output: str) -> ActionReasoningStep:
    """
    Parse an action reasoning step from the LLM output.
    """
    # Weaker LLMs may generate ReActAgent steps whose Action Input are horrible JSON strings.
    # `dirtyjson` is more lenient than `json` in parsing JSON strings.
    import dirtyjson as json

    thought, action, action_input = extract_tool_use(output)
    json_str = extract_json_str(action_input)
    # First we try json, if this fails we use ast
    try:
        action_input_dict = json.loads(json_str)
    except Exception:
        action_input_dict = action_input_parser(json_str)
    return ActionReasoningStep(
        thought=thought, action=action, action_input=action_input_dict
    )


class ReActOutputParser(BaseOutputParser):
    """ReAct Output parser."""

    def parse(self, output: str, is_streaming: bool = False) -> BaseReasoningStep:
        """Parse output from ReAct agent.

        We expect the output to be in one of the following formats:
        1. If the agent need to use a tool to answer the question:
            ```
            Thought: <thought>
            Action: <action>
            Action Input: <action_input>
            ```
        2. If the agent can answer the question without any tools:
            ```
            Thought: <thought>
            Answer: <answer>
            ```
        """
        if "Thought:" not in output:
            # NOTE: handle the case where the agent directly outputs the answer
            # instead of following the thought-answer format
            return ResponseReasoningStep(
                thought="(Implicit) I can answer without any more tools!",
                response=output,
                is_streaming=is_streaming,
            )

        # An "Action" should take priority over an "Answer"
        if "Action:" in output:
            return parse_action_reasoning_step(output)

        if "Answer:" in output:
            thought, answer = extract_final_response(output)
            return ResponseReasoningStep(
                thought=thought, response=answer, is_streaming=is_streaming
            )

        raise ValueError(f"Could not parse output: {output}")

    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError


def escape_newlines_in_json_strings(json_str):
    # Find all strings in the JSON and replace \n within them
    def replace_newline(match):
        # Replace \n with \\n inside the string
        return match.group(0).replace("\n", "\\n")

    # Regular expression to match strings in the JSON
    json_str = re.sub(r"\"(.*?)\"", replace_newline, json_str, flags=re.DOTALL)
    return json_str


class SearchOutputParser(BaseOutputParser):
    """ReAct Output parser."""

    def parse(self, output: str, method: str):
        if method == "explore":
            return self.parse_explore(output)
        elif method == "bug_report":
            return self.parse_bug_report(output)

    def parse_explore(self, output: str) -> Tuple[str, List[SearchActionStep]]:
        """Parse output from Search agent.

        We expect the output to be the following format:
            "observation": "str",
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
        if "obversation_feedback" in output:
            action_list: List[SearchActionStep] = []
            # cast the output to SearchActionStep
            json_str = json.loads(output)
            observation = json_str["obversation_feedback"]
            for action in json_str["new_search_actions"]:
                action_list.append(
                    SearchActionStep(
                        action=action["action"], action_input=action["action_input"]
                    )
                )
            return observation, action_list
        else:
            # raise an error if the output is not in the expected format
            raise ValueError(f"Could not parse search action output: {output}")

    def parse_bug_report(self, output: str) -> List[Dict[str, str]]:
        """
        "bug_locations": [
                {
                    "file": "path/to/file",
                    "function": "function_name",
                    "content": "code_snippet"
                },
                {
                    "file": "path/to/file",
                    "function": "function_name",
                    "content": "code_snippet"
                }
            ]
        """
        if "bug_locations" in output:
            # cast the output to SearchResult
            search_result = escape_newlines_in_json_strings(search_result)
            search_result = json.loads(search_result)

            return search_result
        else:
            # raise an error if the output is not in the expected format
            raise ValueError(f"Could not parse bug report output: {output}")


class ExtractOutputParser(BaseOutputParser):
    """Extractor Agent formatter."""

    def parse(self, output: str, method: str) -> BaseReasoningStep:
        if method == "slice":
            json_obj: Dict = json.loads(output)
            return ExtractSliceStep(
                traceback_warning_log_slice=json_obj["traceback_warning_log_slice"],
                issue_reproducer_slice=json_obj["issue_reproducer_slice"],
                source_code_slice=json_obj["source_code_slice"],
            )
        elif method == "parse":
            json_obj: Dict = json.loads(output)
            code_info_list: List[CodeInfo] = [
                CodeInfo(keyword=x["keyword"], file_path=x["file_path"])
                for x in json_obj["code_info_list"]
            ]
            return ExtractParseStep(code_info_list=code_info_list)
        elif method == "judge":
            json_obj: Dict = json.loads(output)
            return ExtractJudgeStep(
                is_successful=json_obj["is_successful"],
            )
        elif method == "summarize":
            json_obj: Dict = json.loads(output)
            code_info_list: List[CodeInfo] = [
                CodeInfo(keyword=x["keyword"], file_path=x["file_path"])
                for x in json_obj["code_info_list"]
            ]
            return ExtractSummarizeStep(
                summary=json_obj["summary"], code_info_list=code_info_list
            )
        raise NotImplementedError
