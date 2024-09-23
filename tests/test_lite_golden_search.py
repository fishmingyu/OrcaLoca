from Orcar import SearchAgent
import pandas as pd
import csv
import argparse
from Orcar.gen_config import Config, get_llm
from Orcar.environment.benchmark import BenchmarkEnv, get_repo_dir
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.environment.utils import (
    get_container,
    get_logger,
    pause_persistent_container,
    ContainerBash,
)
from Orcar import ExtractAgent
from Orcar.types import ExtractOutput, SearchInput
import json

def load_csv_dataset(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def test_search_agent(instance: str) -> str:
    args_dict = {
        "model": "gpt-4o",
        "image": "sweagent/swe-agent:latest",
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "persistent": True,
        "container_name": "test_0",
        "split": "test",
        "filter_instance": f"^({instance})$",
    }
    args = argparse.Namespace(**args_dict)
    cfg = Config("./key.cfg")
    llm = get_llm(
        model=args.model, api_key=cfg["OPENAI_API_KEY"]
    )
    ctr_name = args.container_name
    docker_ctr_subprocess = get_container(
        ctr_name=ctr_name, image_name=args.image, persistent=args.persistent
    )[0]
    ctr_bash = ContainerBash(ctr_subprocess=docker_ctr_subprocess, ctr_name=ctr_name)

    ds = load_filter_hf_dataset(args)
    env = BenchmarkEnv(args, ctr_bash)
    llm = get_llm(model="gpt-4o")
    inst = ds[0] # only one instance
    env.setup(inst)
    extract_agent = ExtractAgent(llm=llm, env=env, verbose=True)
    try:
        agent_chat_response = extract_agent.chat(json.dumps(dict(inst)))
        extract_output = ExtractOutput.parse_raw(agent_chat_response.response)
        # concat inst["problem_statement"] with the extracted output
        input : SearchInput = SearchInput(problem_statement=inst["problem_statement"], extract_output=extract_output)
        search_agent = SearchAgent(repo_path=env.cache_dir, llm=llm, search_input=input, verbose=False)
        response = search_agent.chat(input.get_content())
    except Exception as e:
        print(f"Error: {e}")
        response = ""

    ctr_bash.ctr_subprocess.stdin.close()
    if args.persistent:
        pause_persistent_container(ctr_bash)
    return response

if __name__ == "__main__":
    instance = "astropy__astropy-12907"
    response = test_search_agent(instance)