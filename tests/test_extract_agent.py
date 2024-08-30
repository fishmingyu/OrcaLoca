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
from Orcar.environment.benchmark import BenchMarkEnv, load_filter_hf_dataset

from Orcar import ExtractAgent
from Orcar.types import ExtractOutput

logger = get_logger("test_extract_agent")

args_dict = {
    "model": "gpt-4o",
    "image": "sweagent/swe-agent:latest",
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "persistent": True,
    "container_name": "test",
    "split": "test",
    # Short Issue Test
    "filter_instance": "^(sympy__sympy-20154)$",
    # Long Issue Test
    # "filter_instance": "^(django__django-15814)$",
    # Multi Issue Test
    # "filter_instance": "^(django__django-15814|psf__requests-2317|django__django-13933|sympy__sympy-20154)$",
}
args = argparse.Namespace(**args_dict)
cfg = Config("./key.cfg")
llm = OpenAI(
    model=args.model, api_key=cfg["OPENAI_API_KEY"], api_base=cfg["OPENAI_API_BASE_URL"]
)


def init_container():
    ctr_name = args.container_name
    docker_ctr_subprocess = get_container(
        ctr_name=ctr_name, image_name=args.image, persistent=args.persistent
    )[0]
    ctr_bash = ContainerBash(ctr_subprocess=docker_ctr_subprocess, ctr_name=ctr_name)

    ds = load_filter_hf_dataset(args)
    return ctr_bash, BenchMarkEnv(args, ctr_bash, ds)


def test_extract_agent():
    ctr_bash, env = init_container()

    agent = ExtractAgent(llm=llm, env=env, verbose=True)
    result_dict = dict()
    for _, inst in env.ds.iterrows():
        agent_chat_response: AgentChatResponse = agent.chat(json.dumps(dict(inst)))
        extract_output = ExtractOutput.parse_raw(agent_chat_response.response)
        result_dict[inst["instance_id"]] = extract_output
        logger.info(extract_output)

    logger.info("Finalizing results:")
    for _, inst in env.ds.iterrows():
        logger.info("-------------------------------------------")
        logger.info(inst["instance_id"])
        logger.info(inst["problem_statement"])
        logger.info(result_dict[inst["instance_id"]])

    ctr_bash.ctr_subprocess.stdin.close()
    if args.persistent:
        pause_persistent_container(ctr_bash)


if __name__ == "__main__":
    test_extract_agent()
