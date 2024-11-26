import argparse
import json
import os
import subprocess
from typing import Any, Dict

from Orcar.environment.benchmark import get_repo_dir, reset_cached_repo
from Orcar.load_cache_dataset import load_filter_hf_dataset_explicit
from Orcar.log_utils import get_logger
from Orcar.search.build_graph import RepoGraph

logger = get_logger(__name__)


def clone_repo(repo: str, to_dir: str):
    if os.path.isdir(to_dir):
        return
    repo_url = f"https://github.com/{repo}.git"
    subprocess.run(["git", "clone", repo_url, to_dir], check=True)


def examine_inst(inst: Dict[str, Any], assets_path: str) -> Dict[str, Any]:
    ret: Dict[str, Any] = dict()
    to_dir = f"{assets_path}/{get_repo_dir(repo=inst['repo'])}"
    clone_repo(repo=inst["repo"], to_dir=to_dir)
    # git reset to commit
    reset_cached_repo(repo_path=to_dir, base_commit=inst["base_commit"])
    # calculate LoC
    lines_of_code = subprocess.run(
        "find . -name '*.py' | xargs cat | wc -l",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        cwd=to_dir,
    ).stdout
    lines_of_code = int(lines_of_code.strip())
    ret["lines_of_code"] = lines_of_code
    # build graph
    graph = RepoGraph(repo_path=to_dir)
    # calculate number of nodes
    ret["number_of_nodes"] = graph.nodes_num
    return ret


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dataset = "SWE-bench_common"
    parser.add_argument(
        "-d",
        "--dataset",
        default=default_dataset,
        help=f"The target dataset (default: {default_dataset})",
    )
    parser.add_argument(
        "-a",
        "--assets_path",
        default="./artifact/assets",
        help=f"The target assets path",
    )
    args = parser.parse_args()
    ds = load_filter_hf_dataset_explicit(
        dataset=args.dataset, filter_instance=".*", split="test"
    )
    output: Dict[str, Any] = dict()
    for i, inst in enumerate(ds):
        logger.info(f"{i+1: 3d}/{len(ds): 3d} Current inst: {inst['instance_id']}")
        inst_output = examine_inst(inst=inst, assets_path=args.assets_path)
        output[inst["instance_id"]] = inst_output
        logger.info(inst_output)
    output_path = f"{args.assets_path}/orcar_insts_stats.json"
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=4)


if __name__ == "__main__":
    main()
