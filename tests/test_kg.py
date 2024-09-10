from Orcar.search import RepoGraph, SearchManager
from llama_index.llms.openai import OpenAI
import argparse
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

def test_build_graph():
    repo_path = "../../django"
    graph_builder = RepoGraph(repo_path=repo_path, save_log=True, log_path="log", build_kg=True)
    # try to search function "add" in the graph
    kg_graph = graph_builder.graph
    root = graph_builder.root_node
    node = graph_builder.get_class_snapshot("ModelChoiceField")
    if node:
        print(f"Snapshot of class ModelChoice   Field: \n {node}")
    else:    
        print("Class snapshot not found")


def test_search_manager():
    repo_path = "../../django"
    search_manager = SearchManager(repo_path=repo_path)
    # try to search function "to_python" in ModelChoiceField class
    file_path, code_snippet = search_manager.search_method_in_class("ModelChoiceField", "to_python")
    print(code_snippet)

def test_local_build_graph():
    repo_graph = "./test_repo"
    graph_builder = RepoGraph(repo_path=repo_graph, save_log=True, log_path="log", build_kg=True)
    # try to search function "add" in the graph
    kg_graph = graph_builder.graph
    root = graph_builder.root_node
    node = graph_builder.get_class_snapshot("B")
    if node:
        print(f"Snapshot of class B: \n {node}")
    else:
        print("Class snapshot not found")
    node_get_sum = graph_builder.dfs_search_method_in_class("B", "sss")
    if node_get_sum:
        print(f"Found the function definition at {node_get_sum}")
    else:
        print("Function definition not found")

def test_env_build_graph():
    logger = get_logger("test_env")

    args_dict = {
        "model": "gpt-4o",
        "image": "sweagent/swe-agent:latest",
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "persistent": True,
        "container_name": "test",
        "split": "test",
        "filter_instance": "^(django__django-13933)$",
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
    for inst in ds:
        env.setup(inst)
        graph_builder = RepoGraph(repo_path=env.cache_dir, save_log=True, log_path="log", build_kg=True)
        node = graph_builder.get_class_snapshot("ModelChoiceField")
        if node:
            print(f"Snapshot of class ModelChoice   Field: \n {node}")
        else:
            print("Class snapshot not found")


if __name__ == "__main__":
    # Example usage
    # test_build_graph()
    # test_search_manager()
    test_local_build_graph()
    # test_env_build_graph()