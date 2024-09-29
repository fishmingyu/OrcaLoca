import argparse

from Orcar import OrcarAgent
from Orcar.gen_config import Config, get_llm
from Orcar.load_cache_dataset import load_filter_hf_dataset

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
    "filter_instance": "^(astropy__astropy-6938)$",
    # Multi Issue Test
    # "filter_instance": "^(pylint-dev__pylint-7080|matplotlib__matplotlib-26020|pytest-dev__pytest-7490)$"
}

def test_agent():
    args = argparse.Namespace(**args_dict)
    cfg = Config("./key.cfg")
    llm = get_llm(model=args.model, api_key=cfg["ANTHROPIC_API_KEY"])
    ds = load_filter_hf_dataset(args)

    #final_stage = 'extract'
    final_stage = 'search'
    agent = OrcarAgent(args=args, llm=llm, final_stage=final_stage)
    agent.set_redirect_log_output(True)
    for inst in ds:
        agent.run(dict(inst))


if __name__ == "__main__":
    test_agent()