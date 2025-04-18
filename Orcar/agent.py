import argparse
import json
import os
import re
import sys
import traceback
from enum import IntEnum
from typing import Any, Dict

from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.llms.llm import LLM

from Orcar.edit_agent import EditAgent
from Orcar.environment.benchmark import BenchmarkEnv, get_repo_dir, reset_cached_repo
from Orcar.environment.utils import (
    ContainerBash,
    get_container,
    pause_persistent_container,
)
from Orcar.log_utils import (
    get_logger,
    set_log_dir,
    switch_log_to_file,
    switch_log_to_stdout,
)
from Orcar.search_agent import SearchAgent
from Orcar.trace_analysis_agent import TraceAnalysisAgent
from Orcar.types import (
    EditInput,
    SearchInput,
    SearchOutput,
    SearchOutputTopK,
    TraceAnalysisOutput,
)


class Stage(IntEnum):
    TRACE_ANALYSIS = 1
    SEARCH = 2
    EDIT = 3


class OrcarAgent:
    """
    Orcar agent worker.
    Environment container lives during agent lifetime;
    Dataset is independent from agent;


    Example call:
    agent = OrcarAgent(args=args, llm=llm, final_stage='Search')
    for inst in ds:
        agent.run(dict(inst))
    """

    def __init__(
        self,
        args: argparse.Namespace,
        llm: LLM,
        final_stage: str,
        current_retry: int = 0,
    ) -> None:
        """
        llm: Should be initiated outside and passed to agent construction
        final_stage: Which stage will agent end at, currently support ["trace_analysis", "search"]
        """
        super().__init__()
        self.logger = get_logger(__name__)
        ctr_name = args.container_name
        docker_ctr_subprocess = get_container(
            ctr_name=ctr_name, image_name=args.image, persistent=args.persistent
        )[0]
        self.llm = llm
        self.persistent = args.persistent
        self.ctr_bash = ContainerBash(
            ctr_subprocess=docker_ctr_subprocess, ctr_name=ctr_name
        )
        self.env = BenchmarkEnv(args, self.ctr_bash)
        self.trace_analysis_agent = TraceAnalysisAgent(
            llm=llm, env=self.env, verbose=False
        )
        self.base_path = self.env.cache_dir
        self.redirect_log: bool = False
        self.output_to_file: bool = True
        self.final_stage = Stage[final_stage.upper()]
        self.output_insts = []
        self.base_log_dir = "./log"
        self.current_retry = current_retry
        if self.current_retry > 0:
            self.base_log_dir += f"_{self.current_retry}"

    def __del__(self) -> None:
        """Pause the container."""
        if hasattr(self, "ctr_bash") and self.ctr_bash.ctr_subprocess.stdin is not None:
            self.ctr_bash.ctr_subprocess.stdin.close()
        if self.persistent:
            pause_persistent_container(self.ctr_bash)

    def set_redirect_log(self, new_value: bool) -> None:
        self.redirect_log = new_value
        if self.redirect_log:
            switch_log_to_file()
        else:
            switch_log_to_stdout()

    def set_output_to_file(self, new_value: bool) -> None:
        self.output_to_file = new_value

    def run_trace_analysis_agent(self) -> TraceAnalysisOutput:
        """Run the trace analysis agent."""
        response: AgentChatResponse = self.trace_analysis_agent.chat(
            json.dumps(dict(self.inst))
        )
        trace_analysis_output = TraceAnalysisOutput.model_validate_json(
            response.response
        )
        self.logger.info("Raw Trace Analysis output:")
        self.logger.info(trace_analysis_output)

        if self.output_to_file:
            trace_analysis_json_obj = json.loads(
                trace_analysis_output.model_dump_json()
            )
            with open(
                f"{self.output_dir}/trace_analyzer_{self.inst_id}.json", "w"
            ) as handle:
                json.dump(trace_analysis_json_obj, handle, indent=4)
            if self.final_stage == Stage.TRACE_ANALYSIS:
                self.output_insts.append(self.inst_id)
        self.logger.info(
            f"Current container subprocess: {self.env.ctr_bash.ctr_subprocess.pid}"
        )

        return trace_analysis_output

    def run_search_agent(
        self, trace_analysis_output: TraceAnalysisOutput
    ) -> SearchOutput:
        """
        Run the search agent.
        It depends on the output of the trace analysis agent.
        """
        search_input = SearchInput(
            problem_statement=self.inst["problem_statement"],
            trace_analysis_output=trace_analysis_output,
        )

        # Create SearchAgent with config_path
        config_path = "search.cfg"
        self.search_agent = SearchAgent(
            repo_path=self.repo_path,
            llm=self.llm,
            search_input=search_input,
            verbose=False,
            config_path=config_path,
        )

        # Get the response from the agent
        search_agent_chat_response: AgentChatResponse = self.search_agent.chat(
            search_input.get_content()
        )

        # Check if the response is in top_k_retrieval_mode format
        response_json = json.loads(search_agent_chat_response.response)

        if "top_files_retrieved" in response_json:
            # Top-K retrieval mode output
            search_output = SearchOutputTopK.model_validate_json(
                search_agent_chat_response.response
            )
            # Convert to regular SearchOutput for backward compatibility
            standard_output = SearchOutput(
                conclusion=search_output.conclusion,
                bug_locations=search_output.bug_locations,
            )
            self.logger.info(
                f"TopK mode output with {len(search_output.top_files_retrieved)} files"
            )
        else:
            # Standard output
            standard_output = SearchOutput.model_validate_json(
                search_agent_chat_response.response
            )
            self.logger.info("Standard output format")

        self.logger.info(standard_output)

        if self.output_to_file:
            # Save the original response (either format) to file
            search_json_obj = json.loads(search_agent_chat_response.response)
            with open(f"{self.output_dir}/searcher_{self.inst_id}.json", "w") as handle:
                json.dump(search_json_obj, handle, indent=4)
            if self.final_stage == Stage.SEARCH:
                self.output_insts.append(self.inst_id)

        return standard_output

    def run_edit_agent(self, search_output: SearchOutput) -> str:
        """
        Run the edit agent.
        It depends on the output of the search agent.
        """
        edit_input = EditInput(
            problem_statement=self.inst["problem_statement"],
            hint=search_output.conclusion,
            bug_locations=search_output.bug_locations,
        )
        self.edit_agent = EditAgent(
            repo_path=self.repo_path,
            llm=self.llm,
            edit_input=edit_input,
            # verbose=False,
        )
        reset_cached_repo(self.repo_path)
        chat_response: AgentChatResponse = self.edit_agent.chat(
            message=edit_input.problem_statement
        )
        edit_output = chat_response.response  # Patch output not finished yet
        self.logger.info(edit_output)
        reset_cached_repo(self.repo_path)

        if self.output_to_file:
            with open(f"{self.output_dir}/editor_{self.inst_id}.patch", "w") as handle:
                handle.write(edit_output)
            if self.final_stage == Stage.EDIT:
                self.output_insts.append(self.inst_id)

        return edit_output

    def run(self, instance: Dict[str, Any]) -> str:
        """Config the output redirection to run agents."""
        self.inst = instance
        self.inst_id = self.inst["instance_id"]
        self.log_dir = f"{self.base_log_dir}/{self.inst_id}"
        self.output_dir = f"./output/{self.inst_id}"
        self.repo_name = get_repo_dir(self.inst["repo"])
        self.repo_path = os.path.join(self.base_path, self.repo_name)
        set_log_dir(self.log_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        if self.redirect_log:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(f"{self.log_dir}/orcar_{self.inst_id}.log", "w") as f:
                sys.stdout = f
                sys.stderr = f
                response = self.run_agents()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        else:
            response = self.run_agents()

        # Redirect log contains format with rich text.
        # Provide a rich-free version for log parsing or less viewing.
        if self.redirect_log:
            with open(f"{self.log_dir}/orcar_{self.inst_id}.log", "r") as f:
                content = f.read()
            content = re.sub(r"\[.*?m", "", content)
            with open(f"{self.log_dir}/orcar_rich_free_{self.inst_id}.log", "w") as f:
                f.write(content)
        return response

    def run_agents(self) -> str:
        """Setup env and run agents."""
        try:
            self.logger.info(
                f"Current container subprocess: {self.env.ctr_bash.ctr_subprocess.pid}"
            )
            self.env.setup(self.inst)
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            return ""

        try:
            trace_analysis_output = self.run_trace_analysis_agent()
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            trace_analysis_output = TraceAnalysisOutput()
        if self.final_stage <= Stage.TRACE_ANALYSIS:
            return trace_analysis_output.model_dump_json(indent=4)

        try:
            search_output = self.run_search_agent(trace_analysis_output)
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            search_output = SearchOutput()
        if self.final_stage <= Stage.SEARCH:
            return search_output.model_dump_json(indent=4)

        try:
            edit_output = self.run_edit_agent(search_output)
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            edit_output = {}
        if self.final_stage <= Stage.EDIT:
            return json.dumps(edit_output, indent=4)

        return f"Invalid final_stage: {self.final_stage}"  # Shouldn't got here

    def __call__(self, text: str) -> str:
        """Call the agent."""
        raise NotImplementedError
