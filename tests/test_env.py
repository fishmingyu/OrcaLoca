import argparse
import os
import re
import sys

from Orcar.environment.benchmark import BenchmarkEnv, get_repo_dir
from Orcar.environment.utils import (
    ContainerBash,
    get_container,
    pause_persistent_container,
)
from Orcar.gen_config import Config, get_llm
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.log_utils import get_logger

logger = get_logger(__name__)

args_dict = {
    "model": "claude-3-5-sonnet-20241022",
    # "model": "gpt-4o",
    "image": "sweagent/swe-agent:latest",
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "persistent": True,
    "container_name": "test_env",
    "split": "test",
    # Short Issue Test
    # "filter_instance": "^(pylint-dev__pylint-7080)$",
    # Multi Issue Test
    # "filter_instance": "^(django__django-15814|psf__requests-2317|django__django-13933|pylint-dev__pylint-7080)$",
    # Env Test
    "filter_instance": ".*",
}
args = argparse.Namespace(**args_dict)
cfg = Config("./key.cfg")
llm = get_llm(model=args.model, api_key=cfg["OPENAI_API_KEY"], max_tokens=4096)
ctr_name = args.container_name
docker_ctr_subprocess = get_container(
    ctr_name=ctr_name, image_name=args.image, persistent=args.persistent
)[0]
ctr_bash = ContainerBash(ctr_subprocess=docker_ctr_subprocess, ctr_name=ctr_name)

ds = load_filter_hf_dataset(args)
env = BenchmarkEnv(args, ctr_bash)


def main():
    # Open the file in write mode
    log_dir = "./log"
    os.makedirs(log_dir, exist_ok=True)
    for _, inst in enumerate(ds):
        # create a new log subdirectory for each instance
        instance_id = inst["instance_id"]
        sub_dir = f"{log_dir}/{instance_id}"
        os.makedirs(sub_dir, exist_ok=True)
        with open(f"{sub_dir}/test_env_{instance_id}.log", "w") as f:
            sys.stdout = f
            env.setup(inst)

    sys.stdout = sys.__stdout__
    for _, inst in enumerate(ds):
        instance_id = inst["instance_id"]
        sub_dir = f"{log_dir}/{instance_id}"
        with open(f"{sub_dir}/test_env_{instance_id}.log", "r") as f:
            content = f.read()
        content = re.sub(r"\[.*?m", "", content)
        with open(f"{sub_dir}/rich_free_{instance_id}.log", "w") as f:
            f.write(content)

    input = "deadbeef\ndeadbeef\ndeadbeef    \n    deadbeef"
    file = "/tmp/test_env.txt"
    env.copy_to_env(input, file)
    output = env.read_text_file(file)
    logger.info(output)
    assert input == output
    logger.info("Walking dir /tmp")
    for root, dirs, files in env.walk("/tmp"):
        logger.info(f"{root=}, {dirs=}, {files=}")

    repo = get_repo_dir(ds[0]["repo"])
    iterate_cnt = 10
    logger.info(f"Walking first {iterate_cnt} items in dir /{repo}")
    iter = env.walk(f"/{repo}")
    for i in range(iterate_cnt):
        try:
            root, dirs, files = next(iter)
            logger.info(f"{i}: {root=}, {dirs=}, {files=}")
        except StopIteration:
            break
    ctr_bash.ctr_subprocess.stdin.close()
    if args.persistent:
        pause_persistent_container(ctr_bash)


if __name__ == "__main__":
    main()
