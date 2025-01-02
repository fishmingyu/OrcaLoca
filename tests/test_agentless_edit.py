import argparse
import json
import os
from typing import Any, Dict, List

from Orcar.environment.benchmark import BenchmarkEnv, get_repo_dir, reset_cached_repo
from Orcar.environment.utils import (
    ContainerBash,
    get_container,
    pause_persistent_container,
)
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.log_utils import get_logger, set_log_dir, switch_log_to_file
from Orcar.search import SearchManager
from Orcar.types import BugLocations

logger = get_logger(__name__)

args_dict = {
    "model": "claude-3-5-sonnet-20241022",
    # "model": "gpt-4o",
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
    "filter_instance": "^(.*)$",
    "file_path_key": "file_name",
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


def gather_search_result():
    args = argparse.Namespace(**args_dict)
    ds = load_filter_hf_dataset(args)

    env_wrapper = EnvWrapper(args)
    jsonl_output = []

    for i, inst in enumerate(ds):
        print(f"({i+1:03d}/{len(ds):03d}) Current inst: {inst['instance_id']}")
        repo_name = get_repo_dir(inst["repo"])
        base_dir = os.path.expanduser("~/.orcar")
        repo_path = os.path.join(base_dir, repo_name)

        # reset to base commit
        base_commit = inst["base_commit"]

        # extract test bug locations
        # open the file ./output/instance_id/search_instance_id.json
        # and extract the bug locations
        search_output_path = (
            f"./output/{inst['instance_id']}/searcher_{inst['instance_id']}.json"
        )
        if not os.path.exists(search_output_path):
            logger.warning(f"Cannot find search output: {search_output_path}")
            ret = {
                "instance_id": inst["instance_id"],
                "found_files": [],
                "found_edit_locs": {},
            }
            logger.info(f"Got json: {ret}")
            jsonl_output.append(ret)
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
        bug_locations: List[BugLocations] = []
        for x in bug_locations_raw:
            if "file_path" not in x and args.file_path_key in x:
                x["file_path"] = x[args.file_path_key]
                x.pop(args.file_path_key)
            bug_locations.append(BugLocations.model_validate(x))
        ret = {}
        ret["instance_id"] = inst["instance_id"]

        found_edit_locs: Dict[str, List[str]] = dict()

        search_manager = SearchManager(repo_path=repo_path)
        exp_length = 0
        for bug in bug_locations:
            if bug.method_name == "":
                loc_info = search_manager._get_exact_loc(
                    f"{bug.file_path}::{bug.class_name}"
                )
            elif bug.class_name == "":
                loc_info = search_manager._get_exact_loc(
                    f"{bug.file_path}::{bug.method_name}"
                )
            else:
                loc_info = search_manager._get_exact_loc(
                    f"{bug.file_path}::{bug.class_name}::{bug.method_name}"
                )
            if loc_info is None:
                logger.warning(f"Cannot find bug code: {bug}")
                continue
            if bug.file_path not in found_edit_locs:
                found_edit_locs[bug.file_path] = []

            """
            if bug.method_name == "":
                bug_str = f"class: {bug.class_name}"
            elif bug.class_name == "":
                if loc_info.type == "global_variable":
                    bug_str = f"variable: {bug.method_name}"
                else:
                    bug_str = f"function: {bug.method_name}"
            else:
                bug_str = f"function: {bug.class_name}.{bug.method_name}"
            """

            # found_edit_locs[bug.file_path].append(bug_str)
            # exp_length += 1

            start_line = int(loc_info.loc.start_line)
            end_line = int(loc_info.loc.end_line)
            for i in range(start_line, end_line + 1):
                found_edit_locs[bug.file_path].append(f"line: {i}")
            exp_length += end_line - start_line + 1

        sum_output_lines = 0
        for _, v in found_edit_locs.items():
            sum_output_lines += len(v)
        if sum_output_lines != exp_length:
            logger.warning(
                f"Output lines mismatch: Got {sum_output_lines} != Exp {exp_length}"
            )

        found_files = sorted(list(set(list(found_edit_locs.keys()))))  # unique
        ret["found_files"] = found_files

        for k, v in found_edit_locs.items():
            found_edit_locs[k] = ["\n".join(v)]
        ret["found_edit_locs"] = found_edit_locs

        logger.info(f"Got json: {ret}")
        jsonl_output.append(ret)

        # reset to base commit
        reset_cached_repo(repo_path, base_commit)
    jsonl_output_file = "loc_merged_0-0_orcar_outputs.jsonl"
    with open(jsonl_output_file, "w") as f:
        for item in jsonl_output:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    gather_search_result()
