from Orcar import SearchAgent
import pandas as pd
import csv
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

def load_csv_dataset(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def test_search_agent(instance: str) -> str:
    args_dict = {
        "model": "gpt-4o",
        "image": "sweagent/swe-agent:latest",
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "persistent": True,
        "container_name": "test",
        "split": "test",
        "filter_instance": f"^({instance})$",
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
    for inst in ds: # only one instance
        env.setup(inst)
        agent = SearchAgent(repo_path=env.cache_dir, llm=llm, verbose=True)
        try:
            response = agent.chat(inst["problem_statement"])
        except Exception as e:
            print(f"Error: {e}")
            response = ""

    ctr_bash.ctr_subprocess.stdin.close()
    if args.persistent:
        pause_persistent_container(ctr_bash)
    return response

if __name__ == "__main__":
    csv_path = "lite_golden_stats.csv"
    df = load_csv_dataset(csv_path)
    save_file = "lite_golden_search_results.csv"

    # Open the file with the 'newline' parameter to prevent extra blank lines
    with open(save_file, "w", newline='', encoding='utf-8') as f:
        # Create a CSV writer object with your desired delimiter
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Write the header row
        writer.writerow(["instance_id", "predicted_patch", "golden_patch"])
        # Iterate over the DataFrame rows
        for i in range(len(df)):
            instance_id = df.iloc[i]["instance_id"]
            response = test_search_agent(instance_id)
            gold_result = df.iloc[i]["parsed_patch"]
            # Write the row to the CSV file
            writer.writerow([instance_id, response, gold_result])
    print(f"Results saved to {save_file}")

