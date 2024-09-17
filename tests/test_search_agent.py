import argparse
import json
import os

from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.llms.openai import OpenAI

from Orcar import ExtractAgent, SearchAgent
from Orcar.environment.benchmark import BenchmarkEnv, get_repo_dir
from Orcar.environment.utils import (
    ContainerBash,
    get_container,
    get_logger,
    pause_persistent_container,
)
from Orcar.key_config import Config
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.types import ExtractOutput

'''
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
        model=args.model,
        api_key=cfg["OPENAI_API_KEY"],
        api_base=cfg["OPENAI_API_BASE_URL"],
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
'''

logger = get_logger("test_search_agent")

args_dict = {
    "model": "gpt-4o",
    "image": "sweagent/swe-agent:latest",
    # "dataset": "princeton-nlp/SWE-bench_Lite",
    "dataset": "SWE-bench_common",
    "persistent": True,
    "container_name": "test",
    "split": "test",
    # Short Issue Test
    "filter_instance": "^(django__django-13933)$",
    # "filter_instance": "^(mwaskom__seaborn-2848)$",
    # Long Issue Test
    # "filter_instance": "^(pylint-dev__pylint-7080)$",
    # Multi Issue Test
    # "filter_instance": "^(django__django-15814|psf__requests-2317|django__django-13933|sympy__sympy-20154)$",
    # Full test
    # "filter_instance": ".*",
    # if django__django-13933 failed, run with 
    # "filter_instance": "^(?!(django__django-13933)$)"
    # if pylint-dev__pylint-7080 failed, 
    # "filter_instance": "^(?!(django__django-13933|pylint-dev__pylint-7080)$)"
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
    return ctr_bash, BenchmarkEnv(args, ctr_bash), ds


def test_search_agent():
    ctr_bash, env, ds = init_container()
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)

    extract_agent = ExtractAgent(llm=llm, env=env, verbose=True)
    
    for inst in ds:
        instance_id = inst['instance_id']

        try:
            env.setup(inst)
        except Exception as e:
            print(f"Error: {e}")
            continue

        try:
            extract_agent_chat_response: AgentChatResponse = extract_agent.chat(json.dumps(dict(inst)))
            extract_output = ExtractOutput.parse_raw(extract_agent_chat_response.response)
            extract_json_obj = json.loads(extract_output.json())
            with open(f'{log_dir}/extractor_{instance_id}.json', 'w') as handle:
                json.dump(extract_json_obj, handle, indent=4)
            logger.info(extract_output)
        except Exception as e:
            extract_json_obj = json.loads('{}')
        
        try:
            # concat inst["problem_statement"] with the extracted output
            search_input = inst["problem_statement"] + "\n" + json.dumps(extract_json_obj)
            search_agent = SearchAgent(repo_path=env.cache_dir, llm=llm, verbose=False)
            search_agent_chat_response: AgentChatResponse = search_agent.chat(search_input)
            logger.info(search_agent_chat_response.response)
            search_output = json.loads(search_agent_chat_response.response)
            search_json_obj = search_output
            with open(f'{log_dir}/searcher_{instance_id}.json', 'w') as handle:
                json.dump(search_json_obj, handle, indent=4)
            logger.info(search_output)
        except Exception as e:
            print(f"Error: {e}")
            continue

    ctr_bash.ctr_subprocess.stdin.close()
    if args.persistent:
        pause_persistent_container(ctr_bash)


if __name__ == "__main__":
    test_search_agent()
