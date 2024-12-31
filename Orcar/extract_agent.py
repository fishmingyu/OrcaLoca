import json
import os
import uuid
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Set, Tuple

from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.agent.types import BaseAgentWorker, Task, TaskStep, TaskStepOutput
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.llms.llm import LLM

from .environment.benchmark import LONG_TIMEOUT, BenchmarkEnv, get_repo_dir
from .formatter import ExtractChatFormatter, TokenCount, TokenCounter
from .log_utils import get_logger
from .output_parser import ExtractOutputParser
from .tracer import gen_tracer_cmd, read_tracer_output
from .tracer_reranker import redirect_filepath_to_cache, rerank_func
from .types import (
    CodeInfo,
    CodeInfoWithClass,
    ExtractJudgeStep,
    ExtractOutput,
    ExtractParseStep,
    ExtractSliceStep,
    ExtractSummarizeStep,
)

logger = get_logger(__name__)

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
        logger.info(
            f"Current extractor container subprocess: {self.env.ctr_bash.ctr_subprocess.pid}"
        )
        self.callback_manager = callback_manager or llm.callback_manager
        self._chat_formatter = ExtractChatFormatter()
        self._output_parser = ExtractOutputParser()
        self._verbose = verbose
        self._token_counter = TokenCounter(llm)

    def chat_with_count(
        self, messages: List[ChatMessage], tag: str, task: Task
    ) -> ChatResponse:
        response, token_cnt = self._token_counter.count_chat(
            messages=messages, llm=self._llm
        )
        task.extra_state["token_cnts"].append((tag, token_cnt))
        return response

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
            "suspicious_code": set(),
            "suspicious_code_from_tracer": list(),
            "suspicious_code_from_tracer_max_size": 5,
            "summary": "",
            "inst": dict(),
            "token_cnts": list(),
            "reproducer_path": "",
            "reproducer_pass": False,
            "reproducer_code": "",
            "reproduce_remaining_trial": 3,
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
            path_ret: PurePath = path_posix_expression
            if (
                len(path_posix_expression.parts) < len(path_windows_expression.parts)
                or path_windows_expression.is_absolute()
            ):
                path_ret = path_windows_expression
            return path_ret

        processed_code_info_list: List[CodeInfo] = []
        # Senstive mechanism:
        # Only care about paths contains certain words
        sensitive_list = ["tests"]  # "tests" should be included
        repo_folder = inst["repo"].split("/")[
            -1
        ]  # for astropy/astropy, "astropy" should be included
        repo_root = "/" + get_repo_dir(inst["repo"])
        if inst["repo"] == "scikit-learn/scikit-learn":
            repo_folder = "sklearn"  # scikit-learn use "sklearn" as import name / main folder name
        sensitive_list += [repo_folder]
        self.env.run(f"cd {repo_root}")

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

            find_output = self.env.run(f"find * -type f -name {path.parts[-1]}")
            candidates = find_output.split("\n")
            output_paths = list(
                filter(lambda x: x.endswith(str(relative_path_suffix)), candidates)
            )
            # It's supposed that keyword at least must show up once in possible file
            output_paths_with_existence = []
            for x in output_paths:
                existence = self.env.run(f"grep -cE '{code_info.keyword}' {x}")
                if int(existence.strip()):
                    output_paths_with_existence.append(x)

            if len(output_paths_with_existence) == 0:
                # path is relevent, but file not found;
                # likely to be a parse error, keep the keyword and drop the path
                processed_code_info = CodeInfo(keyword=code_info.keyword, file_path="")
                processed_code_info_list.append(processed_code_info)
            else:
                for x in output_paths_with_existence:
                    processed_code_info = CodeInfo(
                        keyword=code_info.keyword, file_path=x
                    )
                    processed_code_info_list.append(processed_code_info)

        self.env.run(f"cd -")
        return processed_code_info_list

    def reproduce_issue(self, task: Task) -> str:
        issue_reproducer = task.extra_state["slices"]["reproduce_code_parse"]
        inst = task.extra_state["inst"]
        repo_dir = get_repo_dir(inst["repo"])
        reproducer_path = f"/{repo_dir}/reproducer_{inst['instance_id']}.py"
        task.extra_state["reproducer_path"] = reproducer_path
        task.extra_state["reproducer_code"] = issue_reproducer
        output_path = f"/tmp/tracer_output_{inst['instance_id']}.json"
        self.env.copy_to_env(issue_reproducer, reproducer_path)
        logger.info("Running reproducer...")
        try:
            conda_env = self.env.run("echo $CONDA_DEFAULT_ENV")
            logger.info(f"Conda env: {conda_env}")
            log = self.env.run(
                gen_tracer_cmd(input_path=reproducer_path, output_path=output_path),
                timeout=LONG_TIMEOUT,
            )
        except Exception as e:
            log = str(e)
            logger.warning(f"Reproducer failed: {log}")
            self.env.reset_ctr_bash()
            self.env.setup(inst)
            for cmd in [
                f"cd /{repo_dir}",
                f"conda activate {repo_dir + '__' + inst['version']}",
            ]:
                self.env.run_with_handle(
                    cmd=cmd, err_msg=f"Inst {inst['instance_id']} failed at {cmd=}"
                )
            conda_env = self.env.run("echo $CONDA_DEFAULT_ENV")
            logger.info(f"Conda env: {conda_env}")
            logger.info(
                f"New extractor container subprocess: {self.env.ctr_bash.ctr_subprocess.pid}"
            )
        logger.info(f"Reproducer log:\n{log}")
        return log

    def handle_step_slice(self, step: TaskStep, task: Task) -> List[TaskStep]:
        step_name = step.step_state["name"]
        logger.info(f"Current step: {step_name} in handle_step_slice")

        messages = self._chat_formatter.format(step, task, "slice")
        logger.info(f"{messages}")
        chat_response = self.chat_with_count(
            messages=messages, tag=step_name, task=task
        )
        if chat_response.message.content is None:
            raise ValueError("Got empty message.")
        message_content = chat_response.message.content
        logger.info(f"Chat response: {message_content}")
        slice_step: ExtractSliceStep = self._output_parser.parse(
            message_content, "slice"
        )
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
        # next_step_names.append("summarize")

        return self.gen_next_steps(step, next_step_names)

    def handle_step_parse(self, step: TaskStep, task: Task) -> List[TaskStep]:
        step_name = step.step_state["name"]
        logger.info(f"Current step: {step_name} in handle_step_parse")

        messages = self._chat_formatter.format(step, task, "parse")
        logger.info(f"{messages}")
        chat_response = self.chat_with_count(
            messages=messages, tag=step_name, task=task
        )
        if chat_response.message.content is None:
            raise ValueError("Got empty message.")
        message_content = chat_response.message.content
        logger.info(f"Chat response: {message_content}")
        parse_step: ExtractParseStep = self._output_parser.parse(
            message_content, "parse"
        )

        logger.info(f"Before parse path: {parse_step}")
        parse_step.code_info_list = self.parse_path_in_code_info(
            task.extra_state["inst"], parse_step.code_info_list
        )
        logger.info(f"After parse path: {parse_step}")
        for code_info in parse_step.code_info_list:
            if not code_info.keyword.isidentifier():
                continue
            task.extra_state["suspicious_code"].add(code_info)
        next_step_names: list[str] = []
        return self.gen_next_steps(step, next_step_names)

    def handle_step_judge(self, step: TaskStep, task: Task) -> List[TaskStep]:
        step_name = step.step_state["name"]
        if step_name != "reproduce_judge":
            raise NotImplementedError
        logger.info(f"Current step: {step_name} in handle_step_judge")
        reproduce_log: str = self.reproduce_issue(task=task)
        # TODO: Add iteration
        if not reproduce_log.strip():
            reproduce_log = "**NO LOG WAS GENERATED BY REPRODUCE SNIPPET**"
        task.extra_state["slices"]["reproduce_log_parse"] = reproduce_log
        task.extra_state["parse_type"]["reproduce_log_parse"] = "traceback"

        messages = self._chat_formatter.format(step, task, "judge")
        logger.info(f"{messages}")
        chat_response = self.chat_with_count(
            messages=messages, tag=step_name, task=task
        )
        if chat_response.message.content is None:
            raise ValueError("Got empty message.")
        message_content = chat_response.message.content
        logger.info(f"Chat response: {message_content}")
        judge_step: ExtractJudgeStep = self._output_parser.parse(
            message_content, "judge"
        )
        logger.info(f"{judge_step}")
        task.extra_state["reproduce_remaining_trial"] -= 1
        is_end = judge_step.is_successful or (
            task.extra_state["reproduce_remaining_trial"] == 0
        )
        if judge_step.is_successful:
            task.extra_state["reproducer_pass"] = True

        next_step_names = []
        if not is_end:
            next_step_names.append("reproduce_judge")
            task.extra_state["slices"][
                "reproduce_code_parse"
            ] = judge_step.fixed_reproduce_snippet
        else:
            next_step_names.append("reproduce_code_parse")
            if judge_step.is_successful:
                next_step_names.append("reproduce_log_parse")
                next_step_names.append("reproduce_trace")

        return self.gen_next_steps(step, next_step_names)

    def handle_step_summarize(self, step: TaskStep, task: Task) -> List[TaskStep]:
        step_name = step.step_state["name"]
        logger.info(f"Current step: {step_name} in handle_step_summarize")

        messages = self._chat_formatter.format(step, task, "summarize")
        logger.info(f"{messages}")
        chat_response = self.chat_with_count(
            messages=messages, tag=step_name, task=task
        )
        if chat_response.message.content is None:
            raise ValueError("Got empty message.")
        message_content = chat_response.message.content
        logger.info(f"Chat response: {message_content}")
        summarize_step: ExtractSummarizeStep = self._output_parser.parse(
            message_content, "summarize"
        )

        logger.info(f"{summarize_step.code_info_list}")
        summarize_step.code_info_list = self.parse_path_in_code_info(
            task.extra_state["inst"], summarize_step.code_info_list
        )
        logger.info(f"{summarize_step.code_info_list}")
        for code_info in summarize_step.code_info_list:
            if not code_info.keyword.isidentifier():
                continue
            task.extra_state["suspicious_code"].add(code_info)
        task.extra_state["summary"] = summarize_step.summary

        next_step_names: list[str] = []
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
        output_host_dir = os.path.expanduser(f"~/.orcar/tracer/")
        os.makedirs(output_host_dir, exist_ok=True)
        output_host_path = output_host_dir + f"tracer_output_{instance_id}.json"

        self.env.run(f"ls {output_path}", output_log=True)
        assert os.path.isdir("/tmp")
        self.env.copy_file_from_env(output_path, output_host_path)

        # parse the result
        max_size = task.extra_state["suspicious_code_from_tracer_max_size"]
        sensitivity_list = []
        for c in task.extra_state["suspicious_code"]:
            c: CodeInfo  # path format: 'astropy/modeling/separable.py'
            if not c.file_path:
                sensitivity_list.append(c)
            else:
                full_path = (
                    f"/{get_repo_dir(task.extra_state['inst']['repo'])}/{c.file_path}"
                )
                sensitivity_list.append(
                    CodeInfo(keyword=c.keyword, file_path=full_path)
                )
                # path format: '/astropy__astropy/astropy/modeling/separable.py'
        funcsign_score_list = read_tracer_output(
            output_path=output_host_path, sensitivity_list=sensitivity_list
        )  # Path format: '/astropy__astropy/astropy/modeling/separable.py'
        reproducer_path = task.extra_state["reproducer_path"]
        repo_dir = get_repo_dir(task.extra_state["inst"]["repo"])
        funcsign_score_list = [
            x
            for x in funcsign_score_list
            if (x[0].filename != reproducer_path)
            and (x[0].filename.startswith(f"/{repo_dir}/"))
        ]
        if len(funcsign_score_list) > 5 * max_size:
            logger.info(
                f"Limiting Tracer output from {len(funcsign_score_list)} to {5 * max_size} for reranking"
            )
            funcsign_score_list = funcsign_score_list[
                0 : 5 * max_size
            ]  # limit rerank max size
        else:
            logger.info(f"Tracer output {len(funcsign_score_list)} items for reranking")
        funcsign_score_list = redirect_filepath_to_cache(
            input=funcsign_score_list, cache_dir=self.env.cache_dir
        )  # Path format: '/home/dbmw/.orcar/astropy__astropy/astropy/modeling/separable.py'
        logger.info(f"funcsign_score_list: {funcsign_score_list}")
        funcsign_list, token_cnt = rerank_func(
            input=funcsign_score_list,
            llm=self._llm,
            token_counter=self._token_counter,
            problem_statement=task.extra_state["inst"]["problem_statement"],
        )
        task.extra_state["token_cnts"].append(("tracer_rerank", token_cnt))

        function_list_abs_path = [x.to_codeinfo() for x in funcsign_list]
        function_list = []
        for codeinfo in function_list_abs_path:
            abs_path_parts = codeinfo.file_path.split("/")
            repo_index = len(self.env.cache_dir.split("/")) + 1
            if isinstance(codeinfo, CodeInfoWithClass):
                function_list.append(
                    CodeInfoWithClass(
                        keyword=codeinfo.keyword,
                        file_path="/".join(abs_path_parts[repo_index:]),
                        class_name=codeinfo.class_name,
                    )
                )
            else:
                function_list.append(
                    CodeInfo(
                        keyword=codeinfo.keyword,
                        file_path="/".join(abs_path_parts[repo_index:]),
                    )
                )
        # Path format: 'astropy/modeling/separable.py'

        if len(function_list) > max_size:
            function_list = function_list[0:max_size]
        logger.info(f"After limit size & parse: {function_list}")

        task.extra_state["suspicious_code_from_tracer"] = function_list

        next_step_names: list[str] = []
        os.remove(output_host_path)
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
        ]:
            self.env.run_with_handle(
                cmd=cmd, err_msg=f"Inst {inst['instance_id']} failed at {cmd=}"
            )
        self.env.reset_env_repo(f"/{repo_dir}", inst["base_commit"])

    def gen_output(self, task: Task) -> ExtractOutput:
        suspicious_code_from_tracer: List[CodeInfo] = task.extra_state[
            "suspicious_code_from_tracer"
        ]
        suspicious_keywords_from_tracer = set(
            [code_loc.keyword for code_loc in suspicious_code_from_tracer]
        )
        suspicious_code: Set[CodeInfo] = task.extra_state["suspicious_code"]
        suspicious_code = set(
            [
                code_loc
                for code_loc in suspicious_code
                if code_loc.keyword not in suspicious_keywords_from_tracer
            ]
        )
        related_source_code = ""
        if "source_code_parse" in task.extra_state["slices"]:
            related_source_code = task.extra_state["slices"]["source_code_parse"]
        return ExtractOutput(
            summary=task.extra_state["summary"],
            suspicious_code=list(suspicious_code),
            suspicious_code_from_tracer=suspicious_code_from_tracer,
            related_source_code=related_source_code,
            is_reproduce_pass=task.extra_state["reproducer_pass"],
            reproduce_code=task.extra_state["reproducer_code"],
            env_reproduce_path=task.extra_state["reproducer_path"],
        )

    def _run_step(self, step: TaskStep, task: Task) -> TaskStepOutput:
        task.extra_state["step_done"].remove(step.step_id)

        if step.step_state.get("is_first", False) is True:
            self.handle_first_step(step, task)

        new_steps = self.handle_step(step, task)

        for new_step in new_steps:
            task.extra_state["step_done"].add(new_step.step_id)
        is_done = len(task.extra_state["step_done"]) == 0
        if is_done:
            response = self.gen_output(task)
            agent_response = AgentChatResponse(response=response.model_dump_json())
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
        token_cnts: List[Tuple[str, TokenCount]] = task.extra_state["token_cnts"]
        in_token_cnt = 0
        out_token_cnt = 0
        for tag, token_cnt in token_cnts:
            in_token_cnt += token_cnt.in_token_cnt
            out_token_cnt += token_cnt.out_token_cnt
            logger.info(
                (
                    f"{tag:<25}: "
                    f"in {token_cnt.in_token_cnt:>6} tokens, "
                    f"out {token_cnt.out_token_cnt:>6} tokens"
                )
            )
        logger.info(
            (
                f"{'Total cnt':<25}: "
                f"in {in_token_cnt:>6} tokens, "
                f"out {out_token_cnt:>6} tokens"
            )
        )

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
    extract_output = ExtractOutput.model_validate_json(agent_chat_response.response)
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
