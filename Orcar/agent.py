import argparse
import json
import re
from contextlib import nullcontext, redirect_stdout
from enum import IntEnum
from typing import Any, Dict

from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.llms.llm import LLM

from Orcar.environment.benchmark import BenchmarkEnv, get_repo_dir
from Orcar.environment.utils import (
    ContainerBash,
    get_container,
    get_logger,
    pause_persistent_container,
)
from Orcar.extract_agent import ExtractAgent
from Orcar.search_agent import SearchAgent
from Orcar.types import ExtractOutput, SearchInput

from .environment.utils import ContainerBash


class Stage(IntEnum):
    EXTRACT = 1
    SEARCH = 2


def str2stage(input: str) -> Stage:
    if input.lower() == "extract":
        return Stage.EXTRACT
    if input.lower() == "search":
        return Stage.SEARCH
    raise ValueError(f"str2stage: unrecognized stage {input}")


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

    def __init__(self, args: argparse.Namespace, llm: LLM, final_stage: str) -> None:
        """
        llm: Should be initiated outside and passed to agent construction
        final_stage: Which stage will agent end at, currently support ["extract", "search"]
        """
        super().__init__()
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
        self.extract_agent = ExtractAgent(llm=llm, env=self.env, verbose=True)
        self.base_path = self.env.cache_dir
        self.logger = get_logger("OrcarAgent")
        self.redirect_log_output: bool = False
        self.final_stage = str2stage(final_stage)

    def __del__(self) -> None:
        """Pause the container."""
        self.ctr_bash.ctr_subprocess.stdin.close()
        if self.persistent:
            pause_persistent_container(self.ctr_bash)

    def set_redirect_log_output(self, new_value: bool) -> None:
        self.redirect_log_output = new_value

    def setup_env(self, instance: Dict[str, Any]) -> None:
        self.inst = instance
        self.inst_id = self.inst["instance_id"]
        self.log_dir = f"./log/{self.inst_id}"
        self.env.setup(self.inst)

    def run_extract_agent(self) -> ExtractOutput:
        """Run the extract agent."""
        response: AgentChatResponse = self.extract_agent.chat(
            json.dumps(dict(self.inst))
        )
        extract_output = ExtractOutput.parse_raw(response.response)
        self.logger.info(extract_output)

        extract_json_obj = json.loads(extract_output.json())
        with open(f"{self.log_dir}/extractor_{self.inst_id}.json", "w") as handle:
            json.dump(extract_json_obj, handle, indent=4)

        return extract_output

    def run_search_agent(self, extract_output: ExtractOutput) -> Dict[str, Any]:
        """Run the search agent.
        It depends on the output of the extract agent.
        """
        search_input = SearchInput(
            problem_statement=self.inst["problem_statement"],
            extract_output=extract_output,
        )
        self.search_agent = SearchAgent(
            base_path=self.base_path,
            repo_name=get_repo_dir(self.inst["repo"]),
            llm=self.llm,
            search_input=search_input,
            verbose=False,
        )
        search_agent_chat_response: AgentChatResponse = self.search_agent.chat(
            search_input.get_content()
        )
        self.logger.info(search_agent_chat_response.response)
        search_output = json.loads(search_agent_chat_response.response)
        search_json_obj = search_output
        with open(f"{self.log_dir}/searcher_{self.inst_id}.json", "w") as handle:
            json.dump(search_json_obj, handle, indent=4)
        return search_output

    def run(self, instance: Dict[str, Any]) -> str:
        """Setup env for inst & config the output redirection Run the agent."""
        try:
            self.setup_env(instance)
        except Exception as e:
            print(f"Error: {e}")
            return ""
        # if self.redirect_log_output: log redirected to file
        # else: log printed to normal stdout
        with (
            open(f"{self.log_dir}/orcar_{self.inst_id}.log", "w")
            if self.redirect_log_output
            else nullcontext()
        ) as f:
            with redirect_stdout(f) if self.redirect_log_output else nullcontext():
                response = self.run_agents()
        # Redirect log contains format with rich text.
        # Provide a rich-free version for log parsing or less viewing.
        if self.redirect_log_output:
            with open(f"{self.log_dir}/orcar_{self.inst_id}.log", "r") as f:
                content = f.read()
            content = re.sub(r"\[.*?m", "", content)
            with open(f"{self.log_dir}/orcar_rich_free_{self.inst_id}.log", "w") as f:
                f.write(content)
        return response

    def run_agents(self) -> str:
        """Run agents."""
        try:
            extract_output = self.run_extract_agent()
        except Exception as e:
            print(f"Error: {e}")
            extract_output = ExtractOutput()
        if self.final_stage <= Stage.EXTRACT:
            return extract_output.json(indent=4)

        try:
            search_output = self.run_search_agent(extract_output)
        except Exception as e:
            print(f"Error: {e}")
            search_output = {}
        if self.final_stage <= Stage.SEARCH:
            return json.dumps(search_output, indent=4)
        return f"Invalid final_stage: {self.final_stage}"  # Shouldn't got here

    def __call__(self, text: str) -> str:
        """Call the agent."""
        raise NotImplementedError
