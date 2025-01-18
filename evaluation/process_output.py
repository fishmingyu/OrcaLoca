import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List

from Orcar.environment.benchmark import BenchmarkEnv, get_repo_dir, reset_cached_repo
from Orcar.environment.utils import (
    ContainerBash,
    get_container,
    pause_persistent_container,
)
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.log_utils import get_logger, set_log_dir, switch_log_to_file
from Orcar.types import BugLocations

logger = get_logger(__name__)

args_dict = {
    "model": "claude-3-5-sonnet-20241022",
    "image": "sweagent/swe-agent:latest",
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "persistent": True,
    "container_name": "test_0",
    "split": "test",
    "filter_instance": "^(.*)$",
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


def get_line_range(file_path: str, bug_loc: BugLocations) -> str | None:
    # use ast to analyze the file
    try:
        with open(file_path, "r") as f:
            file_content = f.read()
    except Exception as e:
        logger.warning(f"Cannot read file: {file_path}, {e}")
        return None
    tree = ast.parse(file_content)

    # case 1, method_name and class_name not empty
    if bug_loc.method_name != "" and bug_loc.class_name != "":
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == bug_loc.class_name:
                for subnode in ast.walk(node):
                    if (
                        isinstance(subnode, ast.FunctionDef)
                        and subnode.name == bug_loc.method_name
                    ):
                        return f"[{subnode.lineno}, {subnode.end_lineno}]"

    # case 2, method_name empty, class_name not empty: this is a class
    if bug_loc.method_name == "" and bug_loc.class_name != "":
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == bug_loc.class_name:
                return f"[{node.lineno}, {node.end_lineno}]"

    # case 3, method_name not empty, class_name empty: this is a function
    if bug_loc.method_name != "" and bug_loc.class_name == "":
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == bug_loc.method_name:
                return f"[{node.lineno}, {node.end_lineno}]"

    # case 4, method is not empty, class is empty, but not found, it is a global variable
    # use assign node to find the line number
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == bug_loc.method_name:
                    return f"[{node.lineno}, {node.end_lineno}]"

    # case 5, method is empty, class is not empty, but not found, it is a global variable
    # use assign node to find the line number
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == bug_loc.class_name:
                    return f"[{node.lineno}, {node.end_lineno}]"

    return None


def process_observation(observation: str) -> str:
    # the str begins with <Observation>\n and ends with \n</Observation>
    # we want to remove these tags with regex
    return re.sub(r"<Observation>\n|\n</Observation>", "", observation)


def gather_search_result(run_arg: argparse.Namespace) -> Dict[str, Any]:
    cache_dir = run_arg.cache_dir
    output_dir = run_arg.output_dir
    args = argparse.Namespace(**args_dict)
    ds = load_filter_hf_dataset(args)

    env_wrapper = EnvWrapper(args)
    output_dict = {}

    for i, inst in enumerate(ds):
        print(f"({i+1:03d}/{len(ds):03d}) Current inst: {inst['instance_id']}")
        repo_name = get_repo_dir(inst["repo"])
        base_dir = os.path.expanduser(cache_dir)
        repo_path = os.path.join(base_dir, repo_name)

        # reset to base commit
        base_commit = inst["base_commit"]

        # extract test bug locations
        # open the file $(output_dir)/instance_id/search_instance_id.json
        # and extract the bug locations
        search_output_path = (
            f"{inst['instance_id']}/searcher_{inst['instance_id']}.json"
        )
        search_output_path = os.path.join(output_dir, search_output_path)

        if not os.path.exists(search_output_path):
            logger.warning(f"Cannot find search output: {search_output_path}")
            ret = {
                "instance_id": inst["instance_id"],
                "found_files": [],
                "found_edit_locs": {},
            }
            logger.info(f"Got json: {ret}")
            continue

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

        with open(search_output_path, "r") as f:
            search_output = json.load(f)
        bug_locations_raw: List[Dict[str, Any]] = search_output["bug_locations"]
        logger.info(f"Search bug_locations: {bug_locations_raw}")
        bug_locations = []
        for x in bug_locations_raw:
            bug_loc = BugLocations(
                file_path=x["file_path"],
                class_name=x["class_name"],
                method_name=x["method_name"],
            )
            # filter out test files
            if bug_loc.file_path.startswith("tests/"):
                continue
            joined_file_path = os.path.join(repo_path, x["file_path"])
            line_range = get_line_range(joined_file_path, bug_loc)
            if line_range is None:
                logger.warning(f"Cannot find line range for {bug_loc}")
                continue
            bug_locations.append(
                {
                    "file_path": x["file_path"],
                    "class_name": x["class_name"],
                    "method_name": x["method_name"],
                    "line_range": line_range,
                }
            )
        observation = search_output["conclusion"]
        observation = process_observation(observation)
        output_dict[inst["instance_id"]] = {
            "bug_locations": bug_locations,
            "observation": observation,
        }
    return output_dict


if __name__ == "__main__":
    # set an output directory using argparse
    arg = argparse.ArgumentParser()
    arg.add_argument("--cache_dir", type=str, default="~/.orcar/")
    arg.add_argument("--output_dir", type=str, default="../tests/output/")
    args = arg.parse_args()

    expand_cache_dir = os.path.expanduser(args.cache_dir)
    output_dict = gather_search_result(args)
    # save the dict to a json file
    with open("output.json", "w") as f:
        json.dump(output_dict, f)
