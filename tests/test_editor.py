import argparse
import json
import os
import subprocess

from Orcar import EditAgent
from Orcar.environment.benchmark import get_repo_dir
from Orcar.gen_config import Config, get_llm
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.log_utils import get_logger, set_log_dir, switch_log_to_file
from Orcar.types import EditInput

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
    "filter_instance": (
        "^("
        "astropy__astropy-14995|django__django-12983|django__django-12700"
        "|django__django-13590|django__django-15789|psf__requests-2674"
        "|matplotlib__matplotlib-24149|django__django-14016|matplotlib__matplotlib-23913"
        "|django__django-14999|django__django-11815|django__django-11848"
        "|django__django-15790|astropy__astropy-6938"
        ")$"
    ),
    # "filter_instance": "^(astropy__astropy-12907)$",
}


def reset_cached_repo(repo_path, base_commit):
    proc = subprocess.Popen(
        f"git reset --hard {base_commit}".split(" "),
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.wait()
    proc = subprocess.Popen(
        f"git clean -fdx".split(" "),
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.wait()


def test_agent():
    args = argparse.Namespace(**args_dict)
    cfg = Config("./key.cfg")
    llm = get_llm(model=args.model, api_key=cfg["ANTHROPIC_API_KEY"], max_tokens=4096)
    ds = load_filter_hf_dataset(args)

    for i, inst in enumerate(ds):
        print(f"({i+1:03d}/{len(ds):03d}) Current inst: {inst['instance_id']}")
        set_log_dir(f"./log/{inst['instance_id']}")
        switch_log_to_file()
        repo_name = get_repo_dir(inst["repo"])
        problem_statement = inst["problem_statement"]
        base_dir = os.path.expanduser("~/.orcar")
        repo_path = os.path.join(base_dir, repo_name)

        # reset to base commit
        base_commit = inst["base_commit"]
        reset_cached_repo(repo_path, base_commit)

        # extract test bug locations
        # open the file ./output/instance_id/search_instance_id.json
        # and extract the bug locations
        search_output_path = (
            f"./output/{inst['instance_id']}/searcher_{inst['instance_id']}.json"
        )
        with open(search_output_path, "r") as f:
            search_output = json.load(f)
        bug_locations = search_output["bug_locations"]
        edit_input = EditInput(
            problem_statement=problem_statement, bug_locations=bug_locations
        )
        edit_agent = EditAgent(llm=llm, edit_input=edit_input, repo_path=repo_path)

        chat_response = edit_agent.chat(message=problem_statement)
        edit_output = chat_response.response

        with open(
            f"./output/{inst['instance_id']}/editor_{inst['instance_id']}.patch", "w"
        ) as handle:
            handle.write(edit_output)

        # reset to base commit
        reset_cached_repo(repo_path, base_commit)


if __name__ == "__main__":
    test_agent()
