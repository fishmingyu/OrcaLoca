from .instructor import ReActAgent
from typing import Any, Dict, List, Sequence, Tuple, Union
from llama_index.core.llms.llm import LLM

import time

from .tools import get_llm_compiler_executer, create_tool_list
from .environment.utils import ContainerBash
from Orcar.environment.utils import (
    get_container,
    get_logger,
    pause_persistent_container,
    ContainerBash,
)
from Orcar.search_agent import SearchAgent
from Orcar.environment.benchmark import BenchmarkEnv, get_repo_dir
from Orcar.extract_agent import ExtractAgent
from Orcar.gen_config import Config, get_llm
from Orcar.types import ExtractOutput, SearchInput
import argparse
import json
import os


# construct the Orcar agent

class OrcarAgent:
    """Orcar agent worker."""
    def __init__(self, 
                 args: argparse.Namespace,
                 llm: LLM,
                 instance: Dict[str, Any]
                 ) -> None:
        super().__init__()
        ctr_name = args.container_name
        docker_ctr_subprocess = get_container(
            ctr_name=ctr_name, image_name=args.image, persistent=args.persistent
        )[0]
        self.llm = llm
        self.persistent = args.persistent
        self.ctr_bash = ContainerBash(ctr_subprocess=docker_ctr_subprocess, ctr_name=ctr_name)
        self.env = BenchmarkEnv(args, self.ctr_bash)
        self.env.setup(instance)
        self.extract_agent = ExtractAgent(llm=llm, env=self.env, verbose=True)
        self.repo_name = get_repo_dir(instance["repo"])
        self.base_path = self.env.cache_dir
    
    def run_extract_agent(self, instance: Dict[str, Any]) -> ExtractOutput:
        """Run the extract agent."""
        response = self.extract_agent.chat(json.dumps(dict(instance)))
        extract_output = ExtractOutput.model_validate_json(response.response)
        return extract_output
    
    def run_search_agent(self, instance: Dict[str, Any]) -> str:
        """Run the search agent.
        It depends on the output of the extract agent.
        """
        extract_output = self.run_extract_agent(instance)
        search_input = SearchInput(problem_statement=instance["problem_statement"], extract_output=extract_output)
        self.search_agent = SearchAgent(base_path=self.base_path, repo_name=self.repo_name, llm=self.llm, search_input=search_input, verbose=False)
        response = self.search_agent.chat(search_input.get_content())
        
        return response

    def run(self, instance: Dict[str, Any]) -> str:
        """Run the agent."""
        response = self.run_search_agent(instance)
        self.pause_container()
        return response
    
    def pause_container(self) -> None:
        """Pause the container."""
        self.ctr_bash.ctr_subprocess.stdin.close()
        if self.persistent:
            pause_persistent_container(self.ctr_bash)

    def __call__(self, text: str) -> str:
        """Call the agent."""
        return self.run(text)
