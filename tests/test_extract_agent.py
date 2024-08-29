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

logger = get_logger("test_extract_agent")

args_dict = {
    "model": "gpt-4o",
    "image": "sweagent/swe-agent:latest",
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "persistent": True,
    "container_name": "test",
    "split": "test",
    "filter_instance": "django__django-15814",
}
args = argparse.Namespace(**args_dict)
cfg = Config("./key.cfg")
llm = OpenAI(model=args.model, api_key=cfg['OPENAI_API_KEY'], api_base=cfg['OPENAI_API_BASE_URL'])

def init_container():
    ctr_name = args.container_name
    docker_ctr_subprocess = get_container(
        ctr_name=ctr_name, image_name=args.image, persistent=args.persistent
    )[0]
    ctr_bash = ContainerBash(
        ctr_subprocess=docker_ctr_subprocess, ctr_name=ctr_name
    )

    ds = load_filter_hf_dataset(args)
    return ctr_bash, BenchMarkEnv(args, ctr_bash, ds)

def test_extract_agent():
    ctr_bash, env = init_container()

    agent = ExtractAgent(llm=llm, env=env, verbose=True)
    for _, inst in env.ds.iterrows():
        response: AgentChatResponse = agent.chat(json.dumps(dict(inst)))
        logger.info(json.loads(response.response))

    ctr_bash.ctr_subprocess.stdin.close()
    if args.persistent:
        pause_persistent_container(ctr_bash)

if __name__ == "__main__":
    test_extract_agent()