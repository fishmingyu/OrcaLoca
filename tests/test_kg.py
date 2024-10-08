import argparse
import os

from Orcar.environment.benchmark import BenchmarkEnv
from Orcar.environment.utils import ContainerBash, get_container, get_logger
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.search import RepoGraph, SearchManager

logger = get_logger("test_env")


def test_build_graph():
    repo_path = "../../django"
    graph_builder = RepoGraph(
        repo_path=repo_path, save_log=True, log_path="log", build_kg=True
    )
    # try to search function "add" in the graph
    node = graph_builder.get_class_snapshot("ModelChoiceField")
    if node:
        print(f"Snapshot of class ModelChoice   Field: \n {node}")
    else:
        print("Class snapshot not found")


def test_search_manager():
    repo_path = "../../django"
    search_manager = SearchManager(repo_path=repo_path)
    # try to search function "to_python" in ModelChoiceField class
    file_path, code_snippet = search_manager.search_method_in_class(
        "ModelChoiceField", "to_python"
    )
    print(code_snippet)


def test_local_build_graph():
    repo_graph = "./test_repo"
    graph_builder = RepoGraph(
        repo_path=repo_graph, save_log=True, log_path="log", build_kg=True
    )
    # try to search function "add" in the graph
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
    file_content = graph_builder.dfs_search_file_skeleton("a.py")
    if file_content:
        print(f"File content of a.py: \n {file_content}")
    else:
        print("File content not found")


def test_env_build_graph():

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
    ctr_name = args.container_name
    docker_ctr_subprocess = get_container(
        ctr_name=ctr_name, image_name=args.image, persistent=args.persistent
    )[0]
    ctr_bash = ContainerBash(ctr_subprocess=docker_ctr_subprocess, ctr_name=ctr_name)

    ds = load_filter_hf_dataset(args)
    env = BenchmarkEnv(args, ctr_bash)
    for inst in ds:
        env.setup(inst)
        graph_builder = RepoGraph(
            repo_path=env.cache_dir, save_log=True, log_path="log", build_kg=True
        )
        node = graph_builder.get_class_snapshot("ModelChoiceField")
        if node:
            print(f"Snapshot of class ModelChoice   Field: \n {node}")
        else:
            print("Class snapshot not found")


def test_fitsrec():
    repo_path = "~/.orcar/astropy__astropy"
    # convert the path to absolute
    repo_path = os.path.expanduser(repo_path)
    graph_builder = RepoGraph(
        repo_path=repo_path, save_log=True, log_path="log", build_kg=True
    )
    node = graph_builder.dfs_search_file_skeleton("fitsrec.py")
    if node:
        print(f"Contents of fitsrec.py: \n {node}")
    else:
        print("File contents not found")

    class_ = graph_builder.get_class_snapshot("FITS_rec")
    if class_:
        print(
            f"Snapshot of class FITS_rec: \
            \n {class_}"
        )
    else:
        print("Class snapshot not found")


def test_fitsrec_source_code():
    repo_path = "~/.orcar/astropy__astropy"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    source_code = """
    output_field.replace(encode_ascii('E'),
         encode_ascii('D'))
    """
    line_num = search_manager.search_source_code(
        "astropy/io/fits/fitsrec.py", source_code
    )
    print(line_num)

    print(search_manager.history["search_query"])


def test_search_callable_in_file():
    repo_path = "~/.orcar/astropy__astropy/"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    callable_name = "_scale_back_ascii"
    code_snippet = search_manager.search_callable(
        callable_name, file_path="astropy/io/fits/fitsrec.py"
    )
    print(code_snippet)

    print(search_manager.history["search_query"])


if __name__ == "__main__":
    # Example usage
    # test_build_graph()
    # test_search_manager()
    # test_local_build_graph()
    # test_env_build_graph()
    # test_fitsrec()
    # test_fitsrec_source_code()
    test_search_callable_in_file()
    # print_search_priority()
