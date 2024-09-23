import argparse
import json

from llama_index.core.chat_engine.types import AgentChatResponse

from Orcar import ExtractAgent
from Orcar.environment.benchmark import BenchmarkEnv
from Orcar.environment.utils import (
    ContainerBash,
    get_container,
    get_logger,
    pause_persistent_container,
)
from Orcar.gen_config import Config, get_llm
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.types import ExtractOutput

logger = get_logger("test_extract_agent")

args_dict = {
    "model": "claude-3-5-sonnet-20240620",
    # "model": "gpt-4o",
    "image": "sweagent/swe-agent:latest",
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "persistent": True,
    "container_name": "test_0",
    "split": "test",
    # Short Issue Test
    # "filter_instance": "^(django__django-14999)$",
    # Long Issue Test
    # "filter_instance": "^(astropy__astropy-12907)$",
    "filter_instance": "^(astropy__astropy-6938)$",
    # "filter_instance": "^(django__django-13933)$",
    # "filter_instance": "^(pylint-dev__pylint-7080)$",
    # "filter_instance": "^(matplotlib__matplotlib-26020)$",
    # "filter_instance": "^(pytest-dev__pytest-7490)$",
    # Multi Issue Test
    # "filter_instance": "^(django__django-15814|psf__requests-2317|django__django-13933|sympy__sympy-20154)$",
    # "filter_instance": "^(pylint-dev__pylint-7080|matplotlib__matplotlib-26020|pytest-dev__pytest-7490)$"
}
args = argparse.Namespace(**args_dict)
cfg = Config("./key.cfg")
llm = get_llm(model=args.model, api_key=cfg["OPENAI_API_KEY"], max_tokens=4096)


def init_container():
    ctr_name = args.container_name
    docker_ctr_subprocess = get_container(
        ctr_name=ctr_name, image_name=args.image, persistent=args.persistent
    )[0]
    ctr_bash = ContainerBash(ctr_subprocess=docker_ctr_subprocess, ctr_name=ctr_name)

    ds = load_filter_hf_dataset(args)
    return ctr_bash, BenchmarkEnv(args, ctr_bash), ds


def test_extract_agent():
    ctr_bash, env, ds = init_container()

    agent = ExtractAgent(llm=llm, env=env, verbose=True)
    result_dict = dict()
    for inst in ds:
        env.setup(inst)
        agent_chat_response: AgentChatResponse = agent.chat(json.dumps(dict(inst)))
        extract_output = ExtractOutput.parse_raw(agent_chat_response.response)
        result_dict[inst["instance_id"]] = extract_output
        logger.info(extract_output)

    logger.info("Finalizing results:")
    for inst in ds:
        logger.info("-------------------------------------------")
        logger.info(inst["instance_id"])
        logger.info(inst["problem_statement"])
        logger.info(result_dict[inst["instance_id"]])

    ctr_bash.ctr_subprocess.stdin.close()
    if args.persistent:
        pause_persistent_container(ctr_bash)


if __name__ == "__main__":
    test_extract_agent()
