import json
import argparse
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)


from Orcar.key_config import Config
from Orcar.environment.utils import (
    get_container,
    get_logger,
    pause_persistent_container,
    ContainerBash,
)
from Orcar.environment.benchmark import BenchmarkEnv, load_filter_hf_dataset, get_repo_dir

from Orcar import ExtractAgent
from Orcar.types import ExtractOutput

logger = get_logger("test_env")

args_dict = {
    "model": "gpt-4o",
    "image": "sweagent/swe-agent:latest",
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "persistent": True,
    "container_name": "test",
    "split": "test",
    "filter_instance": "^(pylint-dev__pylint-7080)$",
}
args = argparse.Namespace(**args_dict)
cfg = Config("./key.cfg")
llm = OpenAI(
    model=args.model, api_key=cfg["OPENAI_API_KEY"], api_base=cfg["OPENAI_API_BASE_URL"]
)
ctr_name = args.container_name
docker_ctr_subprocess = get_container(
    ctr_name=ctr_name, image_name=args.image, persistent=args.persistent
)[0]
ctr_bash = ContainerBash(ctr_subprocess=docker_ctr_subprocess, ctr_name=ctr_name)

ds = load_filter_hf_dataset(args)
env = BenchmarkEnv(args, ctr_bash, ds)

def main():
    input = 'deadbeef\ndeadbeef\ndeadbeef    \n    deadbeef'
    file = '/tmp/test_env.txt'
    env.copy_to_env(input, file)
    output = env.read_text_file(file)
    logger.info(output)
    assert input == output
    logger.info("Walking dir /tmp")
    for root, dirs, files in env.walk('/tmp'):
        logger.info(f"{root=}, {dirs=}, {files=}")

    repo = get_repo_dir(env.ds.iloc[0]['repo'])
    iterate_cnt = 10
    logger.info(f"Walking first {iterate_cnt} items in dir /{repo}")
    iter = env.walk(f'/{repo}')
    for i in range(iterate_cnt):
        try:
            root, dirs, files = next(iter)
            logger.info(f"{i}: {root=}, {dirs=}, {files=}")
        except StopIteration:
            break

if __name__ == "__main__":
    main()