import json
import os
import uuid
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.agent.types import BaseAgentWorker, Task, TaskStep, TaskStepOutput
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
)
from llama_index.core.llms.llm import LLM

from .environment.benchmark import BenchmarkEnv, get_repo_dir
from .environment.utils import get_logger
from .formatter import ExtractChatFormatter
from .output_parser import ExtractOutputParser
from .tracer import gen_tracer_cmd, read_tracer_output
from .types import (
    CodeInfo,
    ExtractJudgeStep,
    ExtractOutput,
    ExtractParseStep,
    ExtractSliceStep,
    ExtractSummarizeStep,
)

logger = get_logger("extract_agent")

"""
1. Get name
2. Select prompt from formatter, context from task
3. LLM interaction
4. output parse into step type
5. post-handler (execute reproducer and record log, find relative path from absolute)
6. decide next_steps
step type: slice, parse, judge, summarize? (slices cannot have intersection)
"""

"""
Partial tasks:
1. handle each step in function, move next_steps to each function
2. get inst into worker
3. print prompt for each step
4. parse output for each step
"""

"""
1. slice 
2. reproduce & judge (if has reproduce snippet)
3. parse each part
4. summarize

Need different USER prompt per step; different parse can share description & output format, but have different examples
"""


class ExtractWorker(BaseAgentWorker):
    """Extractor Agent worker."""

    def __init__(
        self,
        llm: LLM,
        env: BenchmarkEnv,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        self._llm = llm
        self.env = env
        self.callback_manager = callback_manager or llm.callback_manager
        self._chat_formatter = ExtractChatFormatter()
        self._output_parser = ExtractOutputParser()
        self._verbose = verbose

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        self.callback_manager = callback_manager

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        sources: List[str] = []
        init_step_id = str(uuid.uuid4())

        # initialize task state
        task_state = {
            "sources": sources,
            "step_done": {init_step_id},
            "slices": dict(),
            "parse_type": dict(),
            "suspicous_code": set(),
            "summary": "",
            "inst": dict(),
        }
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=init_step_id,
            input=task.input,
            step_state={"is_first": True, "name": "slice"},
        )

    def gen_next_steps(
        self, step: TaskStep, next_step_names: List[str]
    ) -> List[TaskStep]:
        return [
            step.get_next_step(
                step_id=str(uuid.uuid4()),
                # NOTE: input is unused
                input=None,
                step_state={"name": next_step_name},
            )
            for next_step_name in next_step_names
        ]

    def parse_path_in_code_info(
        self, inst: Dict[str, Any], related_code_snippets: List[CodeInfo]
    ) -> List[CodeInfo]:

        def cut_since_last_sensitive(target, sensitive):
            # Find the last occurrence of any element from sensitive in target
            for i in range(len(target) - 1, -1, -1):
                if target[i] in sensitive:
                    target = target[i:]
                    break
            return target

        def detect_path_fs(path_str: str):
            # Distinguish file system: Windows or Posix
            path_posix_expression = PurePosixPath(path_str)
            path_windows_expression = PureWindowsPath(path_str)
            path_ret = path_posix_expression
            if (
                len(path_posix_expression.parts) < len(path_windows_expression.parts)
                or path_windows_expression.is_absolute()
            ):
                path_ret = path_windows_expression
            return path_ret

        processed_code_info_list: List[CodeInfo] = []
        sensitive_list = ["tests"]
        repo_folder = inst["repo"].split("/")[-1]
        repo_root = "/" + get_repo_dir(inst["repo"])
        if inst["repo"] == "scikit-learn/scikit-learn":
            repo_folder = "sklearn"
        sensitive_list += [repo_folder]

        for code_info in related_code_snippets:
            if code_info.file_path == "":
                # no path found, stay as is
                processed_code_info_list.append(code_info)
                continue

            path = detect_path_fs(code_info.file_path)

            if not any([p in path.parts for p in sensitive_list]):
                # irrelevant path, drop
                continue
            relative_path_suffix = PurePath(
                *cut_since_last_sensitive(path.parts, sensitive_list)
            )

            # if not (path.is_absolute() or path.parts[0] == "~"):
            #    # relative path, stay as is
            #    processed_code_info_list.append(code_info)
            #    continue

            find_output = self.env.run(f"find {repo_root} -name {path.parts[-1]}")
            candidates = find_output.split("\n")
            output_paths = list(
                filter(lambda x: x.endswith(str(relative_path_suffix)), candidates)
            )
            for x in output_paths:
                processed_code_info = CodeInfo(keyword=code_info.keyword, file_path=x)
                processed_code_info_list.append(processed_code_info)
            if len(output_paths) == 0:
                # path is relevent, but file not found;
                # likely to be a parse error, keep the keyword and drop the path
                processed_code_info = CodeInfo(keyword=code_info.keyword, file_path="")
                processed_code_info_list.append(processed_code_info)

        return processed_code_info_list

    def reproduce_issue(self, issue_reproducer: str, inst: Dict[str, Any]) -> str:
        repo_dir = get_repo_dir(inst["repo"])
        reproducer_path = f"/{repo_dir}/reproducer_{inst['instance_id']}.py"
        output_path = f"/tmp/tracer_output_{inst['instance_id']}.json"
        self.env.copy_to_env(issue_reproducer, reproducer_path)
        logger.info("Running reproducer...")
        log = self.env.run(
            gen_tracer_cmd(input_path=reproducer_path, output_path=output_path),
            output_log=True,
        )
        return log

    def handle_step_slice(self, step: TaskStep, task: Task) -> List[TaskStep]:
        step_name = step.step_state["name"]
        logger.info(f"Current step: {step_name} in handle_step_slice")

        # TODO: extract into a function?
        prompt = self._chat_formatter.format(step, task, "slice")
        logger.info(f"{prompt}")
        chat_response = self._llm.chat(messages=prompt)
        if chat_response.message.content is None:
            raise ValueError("Got empty message.")
        message_content = chat_response.message.content
        logger.info(f"Chat response: {message_content}")
        slice_step = self._output_parser.parse(message_content, "slice")
        logger.info(f"{slice_step}")

        next_step_names = []
        if slice_step.traceback_warning_log_slice:
            next_step_names.append("traceback_parse")
            task.extra_state["slices"][
                "traceback_parse"
            ] = slice_step.traceback_warning_log_slice
            task.extra_state["parse_type"]["traceback_parse"] = "traceback"
        if slice_step.issue_reproducer_slice:
            next_step_names.append("reproduce_judge")
            task.extra_state["slices"][
                "reproduce_code_parse"
            ] = slice_step.issue_reproducer_slice
            task.extra_state["parse_type"]["reproduce_code_parse"] = "code"
        if slice_step.source_code_slice:
            next_step_names.append("source_code_parse")
            task.extra_state["slices"][
                "source_code_parse"
            ] = slice_step.source_code_slice
            task.extra_state["parse_type"]["source_code_parse"] = "code"
        if slice_step.natural_language_description_slice:
            next_step_names.append("NL_parse")
            task.extra_state["slices"][
                "NL_parse"
            ] = slice_step.natural_language_description_slice
            task.extra_state["parse_type"]["NL_parse"] = "NL"
        next_step_names.append("summarize")

        return self.gen_next_steps(step, next_step_names)

    def handle_step_parse(self, step: TaskStep, task: Task) -> List[TaskStep]:
        step_name = step.step_state["name"]
        logger.info(f"Current step: {step_name} in handle_step_parse")

        prompt = self._chat_formatter.format(step, task, "parse")
        logger.info(f"{prompt}")
        chat_response = self._llm.chat(messages=prompt)
        if chat_response.message.content is None:
            raise ValueError("Got empty message.")
        message_content = chat_response.message.content
        logger.info(f"Chat response: {message_content}")
        parse_step: ExtractParseStep = self._output_parser.parse(
            message_content, "parse"
        )

        logger.info(f"{parse_step}")
        parse_step.code_info_list = self.parse_path_in_code_info(
            task.extra_state["inst"], parse_step.code_info_list
        )
        logger.info(f"{parse_step}")
        task.extra_state["suspicous_code"].update(parse_step.code_info_list)
        next_step_names = []
        return self.gen_next_steps(step, next_step_names)

    def handle_step_judge(self, step: TaskStep, task: Task) -> List[TaskStep]:
        step_name = step.step_state["name"]
        if step_name != "reproduce_judge":
            raise NotImplementedError
        logger.info(f"Current step: {step_name} in handle_step_judge")
        task.extra_state["slices"]["reproduce_log_parse"] = self.reproduce_issue(
            issue_reproducer=task.extra_state["slices"]["reproduce_code_parse"],
            inst=task.extra_state["inst"],
        )
        task.extra_state["parse_type"]["reproduce_log_parse"] = "traceback"

        prompt = self._chat_formatter.format(step, task, "judge")
        logger.info(f"{prompt}")
        chat_response = self._llm.chat(messages=prompt)
        if chat_response.message.content is None:
            raise ValueError("Got empty message.")
        message_content = chat_response.message.content
        logger.info(f"Chat response: {message_content}")
        judge_step: ExtractJudgeStep = self._output_parser.parse(
            message_content, "judge"
        )
        logger.info(f"{judge_step}")

        next_step_names = []
        if judge_step.is_successful:
            next_step_names.append("reproduce_log_parse")
            next_step_names.append("reproduce_trace")
        else:
            next_step_names.append("reproduce_code_parse")
        return self.gen_next_steps(step, next_step_names)

    def handle_step_summarize(self, step: TaskStep, task: Task) -> List[TaskStep]:
        step_name = step.step_state["name"]
        logger.info(f"Current step: {step_name} in handle_step_summarize")

        prompt = self._chat_formatter.format(step, task, "summarize")
        logger.info(f"{prompt}")
        chat_response = self._llm.chat(messages=prompt)
        if chat_response.message.content is None:
            raise ValueError("Got empty message.")
        message_content = chat_response.message.content
        logger.info(f"Chat response: {message_content}")
        summarize_step: ExtractSummarizeStep = self._output_parser.parse(
            message_content, "summarize"
        )

        task.extra_state["summary"] = summarize_step.summary
        next_step_names = []
        return self.gen_next_steps(step, next_step_names)

    def handle_step_trace(self, step: TaskStep, task: Task) -> List[TaskStep]:
        step_name = step.step_state["name"]
        if step_name != "reproduce_trace":
            raise NotImplementedError
        logger.info(f"Current step: {step_name} in handle_step_trace")

        # Get instance ID
        instance_id = task.extra_state["inst"]["instance_id"]

        # docker cp the result out
        output_path = f"/tmp/tracer_output_{instance_id}.json"
        self.env.run(f"ls {output_path}", output_log=True)
        assert os.path.isdir("/tmp")
        self.env.copy_file_from_env(output_path, output_path)

        # parse the result
        sensitivity_list = [
            code_info.keyword for code_info in task.extra_state["suspicous_code"]
        ]
        logger.info(f"sensitivity_list: {sensitivity_list}")
        function_list = read_tracer_output(
            output_path=output_path, sensitivity_list=sensitivity_list
        )
        function_list = self.parse_path_in_code_info(
            task.extra_state["inst"], function_list
        )
        logger.info(f"function_list: {function_list}")
        task.extra_state["suspicous_code"].update(function_list)

        next_step_names = []
        return self.gen_next_steps(step, next_step_names)

    def handle_step(self, step: TaskStep, task: Task) -> List[TaskStep]:
        step_name = step.step_state["name"]
        if "slice" in step_name:
            return self.handle_step_slice(step, task)
        elif "parse" in step_name:
            return self.handle_step_parse(step, task)
        elif "judge" in step_name:
            return self.handle_step_judge(step, task)
        elif "summarize" in step_name:
            return self.handle_step_summarize(step, task)
        elif "trace" in step_name:
            return self.handle_step_trace(step, task)
        raise ValueError(
            f"ExtractWorker.handle_step: Cannot recognize step name {step_name}"
        )

    def handle_first_step(self, step: TaskStep, task: Task) -> None:
        inst = json.loads(step.input)
        task.extra_state["inst"] = inst
        repo_dir = get_repo_dir(inst["repo"])
        for cmd in [
            f"cd /{repo_dir}",
            f"conda activate {repo_dir + '__' + inst['version']}",
            f"git reset --hard {inst['base_commit']}",
        ]:
            self.env.run_with_handle(
                cmd=cmd, err_msg=f"Inst {inst['instance_id']} failed at {cmd=}"
            )

    def _run_step(self, step: TaskStep, task: Task) -> TaskStepOutput:
        task.extra_state["step_done"].remove(step.step_id)

        if step.step_state.get("is_first", False) == True:
            self.handle_first_step(step, task)

        new_steps = self.handle_step(step, task)

        for new_step in new_steps:
            task.extra_state["step_done"].add(new_step.step_id)
        is_done = len(task.extra_state["step_done"]) == 0
        if is_done:
            response = ExtractOutput(
                summary=task.extra_state["summary"],
                suspicous_code=list(task.extra_state["suspicous_code"]),
            )
            agent_response = AgentChatResponse(response=response.json())
        else:
            agent_response = AgentChatResponse(response="")

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""
        # TODO
        pass

    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        return self._run_step(step, task)

    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        raise NotImplementedError

    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        raise NotImplementedError

    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        raise NotImplementedError


class ExtractAgent(AgentRunner):
    """
    Extractor Agent. Response type: ExtractOutput

    Calling example:
    agent = ExtractAgent(llm=llm, env=env, verbose=True)
    agent_chat_response: AgentChatResponse = agent.chat(input)

    Response parse:
    extract_output = ExtractOutput.parse_raw(agent_chat_response.response)
    """

    def __init__(
        self,
        llm: LLM,
        env: BenchmarkEnv,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        callback_manager = callback_manager or llm.callback_manager

        step_engine = ExtractWorker(
            llm=llm,
            env=env,
            callback_manager=callback_manager,
            verbose=verbose,
        )
        if callback_manager is not None:
            llm.callback_manager = callback_manager

        super().__init__(
            step_engine,
            llm=llm,
            callback_manager=callback_manager,
            verbose=verbose,
        )
