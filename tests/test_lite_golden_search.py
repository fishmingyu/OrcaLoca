from Orcar import SearchAgent
import pandas as pd
import csv
import argparse
from Orcar.gen_config import Config, get_llm
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar import OrcarAgent

def load_csv_dataset(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def test_search_agent(instance: str) -> str:
    args_dict = {
        "model": "claude-3-5-sonnet-20240620",
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
        model=args.model, api_key=cfg["ANTHROPIC_API_KEY"]
    )
    ds = load_filter_hf_dataset(args)
    inst = ds[0] # only one instance
    agent = OrcarAgent(args, llm, inst)
    agent.run(inst)


if __name__ == "__main__":
    instance = "astropy__astropy-12907"
    response = test_search_agent(instance)