import argparse
import json
import os

from Orcar import EditAgent
from Orcar.environment.benchmark import BenchmarkEnv, get_repo_dir, reset_cached_repo
from Orcar.environment.utils import (
    ContainerBash,
    get_container,
    pause_persistent_container,
)
from Orcar.gen_config import Config, get_llm
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.log_utils import get_logger, set_log_dir, switch_log_to_file
from Orcar.types import EditInput

logger = get_logger(__name__)

args_dict = {
    # "model": "claude-3-5-sonnet-20241022",
    # "model": "gpt-4o",
    "model": "gemini-2.0-pro-exp-02-05",
    "image": "sweagent/swe-agent:latest",
    # "dataset": "SWE-bench_common",
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "persistent": True,
    "container_name": "test_0",
    "split": "test",
    # Short Issue Test
    # "filter_instance": "^(matplotlib__matplotlib-23314)$",
    # "filter_instance": "^(django__django-15814)$",
    # "filter_instance": "^(astropy__astropy-14182)$",
    # Long Issue Test
    # "filter_instance": (
    #    "^("
    #    "astropy__astropy-14995|django__django-12983|django__django-12700"
    #    "|django__django-13590|django__django-15789|psf__requests-2674"
    #    "|matplotlib__matplotlib-24149|django__django-14016|matplotlib__matplotlib-23913"
    #   "|django__django-14999|django__django-11815|django__django-11848"
    #    "|django__django-15790|astropy__astropy-6938"
    #    ")$"
    # ),
    "filter_instance": "^(astropy__astropy-6938)$",
}


class EnvWrapper:
    def __init__(self, args):
        ctr_name = args.container_name
        docker_ctr_subprocess = get_container(
            ctr_name=ctr_name, image_name=args.image, persistent=args.persistent
        )[0]
        self.ctr_bash = ContainerBash(
            ctr_subprocess=docker_ctr_subprocess, ctr_name=ctr_name
        )
        self.env = BenchmarkEnv(args, self.ctr_bash)
        self.persistent = args.persistent

    def setup(self, inst, switch_log=True):
        set_log_dir(f"./log/{inst['instance_id']}")
        if switch_log:
            switch_log_to_file()
        self.env.setup(inst)
        repo_dir = get_repo_dir(inst["repo"])
        self.env.run(
            f"conda activate {repo_dir + '__' + inst['version']}", output_log=True
        )

    def __del__(self) -> None:
        """Pause the container."""
        if self.ctr_bash.ctr_subprocess.stdin is not None:
            self.ctr_bash.ctr_subprocess.stdin.close()
        if self.persistent:
            pause_persistent_container(self.ctr_bash)


def test_agent():
    args = argparse.Namespace(**args_dict)
    cfg = Config("../key.cfg")
    llm = get_llm(model=args.model, max_tokens=4096, orcar_config=cfg)
    ds = load_filter_hf_dataset(args)

    env_wrapper = EnvWrapper(args)

    # read json from output.json
    output_json = json.load(open("./output.json", "r"))
    # read dependency from dependency.json
    dependency_json = json.load(open("./dependency_output.json", "r"))

    for i, inst in enumerate(ds):
        print(f"({i+1:03d}/{len(ds):03d}) Current inst: {inst['instance_id']}")
        env_wrapper.setup(inst, switch_log=False)

        repo_name = get_repo_dir(inst["repo"])
        problem_statement = inst["problem_statement"]
        base_dir = os.path.expanduser("~/.orcar")
        repo_path = os.path.join(base_dir, repo_name)

        # reset to base commit
        base_commit = inst["base_commit"]
        reset_cached_repo(repo_path, base_commit)

        # read the corresponding output from output.json
        search_output = output_json[inst["instance_id"]]
        dependency = dependency_json[inst["instance_id"]]

        print(f"Output: {search_output}")
        print(f"Dependency: {dependency}")

        # extract test bug locations
        bug_locations = search_output["bug_locations"]
        edit_input = EditInput(
            problem_statement=problem_statement,
            hint=search_output["observation"],
            bug_locations=bug_locations,
            dependency=dependency,
        )
        edit_agent = EditAgent(llm=llm, edit_input=edit_input, repo_path=repo_path)

        chat_response = edit_agent.chat(message=problem_statement)
        edit_output = chat_response.response
        print(f"Edit Output: {edit_output}")

        # with open(
        #     f"./output/{inst['instance_id']}/editor_{inst['instance_id']}.patch", "w"
        # ) as handle:
        #     handle.write(edit_output)

        # reset to base commit
        reset_cached_repo(repo_path, base_commit)


if __name__ == "__main__":
    test_agent()
