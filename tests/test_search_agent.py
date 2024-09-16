from Orcar import SearchAgent
from llama_index.llms.openai import OpenAI
import argparse
from Orcar.key_config import Config
from Orcar.environment.benchmark import BenchmarkEnv, get_repo_dir
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.environment.utils import (
    get_container,
    get_logger,
    pause_persistent_container,
    ContainerBash,
)
import json
from Orcar.environment.benchmark import BenchmarkEnv
from Orcar.load_cache_dataset import load_filter_hf_dataset


from Orcar import ExtractAgent
from Orcar.types import ExtractOutput

def test_search_agent():
    args_dict = {
        "model": "gpt-4o",
        "image": "sweagent/swe-agent:latest",
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "persistent": True,
        "container_name": "test_0",
        "split": "test",
        "filter_instance": "^(astropy__astropy-12907)$",
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
    env = BenchmarkEnv(args, ctr_bash)
    llm = OpenAI(model="gpt-4o")
    for inst in ds:
        env.setup(inst)
        agent = ExtractAgent(llm=llm, env=env, verbose=True)
        agent_chat_response = agent.chat(json.dumps(dict(inst)))
        extract_output = ExtractOutput.parse_raw(agent_chat_response.response)
        agent = SearchAgent(repo_path=env.cache_dir, llm=llm, verbose=False)
        # concat inst["problem_statement"] with the extracted output
        input = inst["problem_statement"] + "\n" + str(extract_output)
        response = agent.chat(input)
        print(response)

if __name__ == "__main__":
    test_search_agent()