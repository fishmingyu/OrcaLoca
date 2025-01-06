import argparse
import json
import os
from typing import Any, Dict

from Orcar.environment.benchmark import BenchmarkEnv, get_repo_dir, reset_cached_repo
from Orcar.environment.utils import (
    ContainerBash,
    get_container,
    pause_persistent_container,
)
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.log_utils import get_logger, set_log_dir, switch_log_to_file
from Orcar.search import SearchManager

logger = get_logger(__name__)

args_dict = {
    "model": "claude-3-5-sonnet-20241022",
    "image": "sweagent/swe-agent:latest",
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "persistent": True,
    "container_name": "test_0",
    "split": "test",
    "filter_instance": "^(.*)$",
    # "filter_instance": "^(astropy__astropy-12907)$",
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

    def cache(self, inst, switch_log=True):
        set_log_dir(f"./log/{inst['instance_id']}")
        if switch_log:
            switch_log_to_file()
        self.env.clone_repo(inst)
        self.env.cache_repo_to_host(inst)

    def __del__(self) -> None:
        """Pause the container."""
        if self.ctr_bash.ctr_subprocess.stdin is not None:
            self.ctr_bash.ctr_subprocess.stdin.close()
        if self.persistent:
            pause_persistent_container(self.ctr_bash)


def get_search_query(file_path: str, class_name: str, method_name: str) -> str | None:
    # four cases:
    # 1. class_name is "" and method_name is ""
    # 2. class_name is not "" and method_name is ""
    # 3. class_name is not "" and method_name is not ""
    # 4. class_name is "" and method_name is not ""
    if class_name == "" and method_name == "":
        return f"{file_path}"
    elif class_name != "" and method_name == "":  # this is a class
        return f"{file_path}::{class_name}"
    elif class_name != "" and method_name != "":  # this is a method
        return f"{file_path}::{class_name}::{method_name}"
    elif class_name == "" and method_name != "":  # this is a function
        return f"{file_path}::{method_name}"
    else:
        return None


def get_dependency(search_input: Dict[str, Any]) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    ds = load_filter_hf_dataset(args)

    env_wrapper = EnvWrapper(args)
    output_dict = {}

    for i, inst in enumerate(ds):
        print(f"({i+1:03d}/{len(ds):03d}) Current inst: {inst['instance_id']}")

        repo_name = get_repo_dir(inst["repo"])
        base_dir = os.path.expanduser("~/.orcar")
        repo_path = os.path.join(base_dir, repo_name)
        base_commit = inst["base_commit"]

        reset_succeeded = False
        if os.path.exists(repo_path):
            try:
                reset_cached_repo(repo_path, base_commit)
                reset_succeeded = True
            except Exception as e:
                logger.warning(f"Reset failed: {e}")
        if not reset_succeeded:
            env_wrapper.cache(inst, switch_log=False)
            reset_cached_repo(repo_path, base_commit)

        # set up the search manager
        search_manager = SearchManager(repo_path)

        # get the search_input (per instance)
        instance_info = search_input.get(inst["instance_id"])
        if instance_info is None:
            logger.warning(f"Instance {inst['instance_id']} not found in search input")
            continue

        # get bug locations
        bug_locs = instance_info["bug_locations"]
        search_dep_locs = []
        for bug_loc in bug_locs:
            file_path = bug_loc["file_path"]
            class_name = bug_loc["class_name"]
            method_name = bug_loc["method_name"]
            search_query = get_search_query(file_path, class_name, method_name)
            # use _get_exact_loc
            exact_loc = search_manager._get_dependency(search_query)
            if exact_loc is None:
                logger.warning(f"Cannot find exact location for {search_query}")
                continue
            for dep in exact_loc:
                dep_file_path = dep.file_name
                start_line = dep.start_line
                end_line = dep.end_line
                line_range = f"[{start_line}, {end_line}]"
                dep_dict = {
                    "file_path": dep_file_path,
                    "name": dep.node_name,
                    "line_range": line_range,
                }
                # check not appears in dep_locs
                if dep_dict in search_dep_locs:
                    continue
                print(f"Dependency: {dep_dict}")
                search_dep_locs.append(dep_dict)

        output_dict[inst["instance_id"]] = search_dep_locs
    return output_dict


if __name__ == "__main__":
    cache_dir = "~/.orcar/"
    expand_cache_dir = os.path.expanduser(cache_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--search_input",
        default="output.json",
        help=f"The search input file",
    )
    args = parser.parse_args()
    # open the search input file
    with open(args.search_input, "r") as f:
        search_input = json.load(f)

    output_dict = get_dependency(search_input)
    # write the output to a file
    with open("dependency_output.json", "w") as f:
        json.dump(output_dict, f, indent=4)
