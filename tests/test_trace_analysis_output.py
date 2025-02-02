import argparse
import json
import os

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
    "dataset": "SWE-bench_common",
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


def parse_trace_analyzer_output():
    args = argparse.Namespace(**args_dict)
    ds = load_filter_hf_dataset(args)

    env_wrapper = EnvWrapper(args)

    for i, inst in enumerate(ds):
        print(f"({i+1:03d}/{len(ds):03d}) Current inst: {inst['instance_id']}")
        repo_name = get_repo_dir(inst["repo"])
        base_dir = os.path.expanduser("~/.orcar")
        repo_path = os.path.join(base_dir, repo_name)

        # reset to base commit
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

        # extract test bug locations
        # open the file ./output/instance_id/search_instance_id.json
        # and extract the bug locations
        trace_analyzer_output_path = (
            f"./output/{inst['instance_id']}/trace_analyzer_{inst['instance_id']}.json"
        )

        if not os.path.exists(trace_analyzer_output_path):
            logger.warning(f"Cannot find search output: {trace_analyzer_output_path}")
            continue

        search_manager = SearchManager(repo_path=repo_path)

        with open(trace_analyzer_output_path, "r") as handle:
            trace_analyzer_output = json.load(handle)
        logger.info(f"Extract output: {trace_analyzer_output}")

        ret = {"conclusion": "", "bug_locations": []}
        for code in trace_analyzer_output["suspicious_code"]:
            ret["bug_locations"].extend(
                search_manager._get_bug_location(
                    code["keyword"], code["file_path"] if code["file_path"] else None
                )["bug_locations"]
            )
        logger.info(f"Got json: {ret}")

        search_output_path = f"./output_extract_search_style/{inst['instance_id']}/searcher_{inst['instance_id']}.json"
        if not os.path.exists(os.path.dirname(search_output_path)):
            os.makedirs(os.path.dirname(search_output_path))
        with open(search_output_path, "w") as f:
            json.dump(ret, f, indent=4)


if __name__ == "__main__":
    parse_trace_analyzer_output()
