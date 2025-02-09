import argparse
import os

from Orcar.editor import Editor
from Orcar.environment.benchmark import BenchmarkEnv
from Orcar.environment.utils import ContainerBash, get_container
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.log_utils import get_logger
from Orcar.search import RepoGraph, SearchManager

logger = get_logger(__name__)


def test_build_graph():
    repo_path = "~/.orcar/django__django"
    expand_repo_path = os.path.expanduser(repo_path)
    graph_builder = RepoGraph(
        repo_path=expand_repo_path, save_log=True, log_path="log", build_kg=True
    )
    # try to search function "add" in the graph
    node = graph_builder.dfs_get_class_snapshot("ModelChoiceField")
    if node:
        print(f"Snapshot of class ModelChoice   Field: \n {node}")
    else:
        print("Class snapshot not found")


def test_search_manager():
    repo_path = "~/.orcar/django__django"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    # try to search function "to_python" in ModelChoiceField class
    search_class_res = search_manager.search_class("ModelChoiceField")
    print(search_class_res)
    dataframe = search_manager.get_frame_from_history(
        action="search_class", input="ModelChoiceField"
    )
    file_path = dataframe["file_path"]

    # try search method in class
    exact_search_result = search_manager.search_method_in_class(
        class_name="ModelChoiceField", method_name="to_python", file_path=file_path
    )
    print(exact_search_result)

    # try search method in class without specifying file path
    exact_search_result = search_manager.search_method_in_class(
        class_name="ModelChoiceField", method_name="to_python"
    )
    print(exact_search_result)


def test_exact_search():
    repo_path = "~/.orcar/django__django"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    # try to search function "to_python" in ModelChoiceField class
    exact_search_result = search_manager.exact_search(
        query="ModelChoiceField",
        file_path="django/forms/models.py",
        containing_class="ModelChoiceField",
    )
    print(exact_search_result)
    exact_search_result = search_manager.exact_search(
        query="ASCIIUsernameValidator", file_path="django/contrib/auth/validators.py"
    )
    print(exact_search_result)


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
        node = graph_builder.dfs_get_class_snapshot("ModelChoiceField")
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

    class_ = graph_builder.dfs_get_class_snapshot("FITS_rec")
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
    exist = search_manager.get_node_existence("astropy/io/fits/fitsrec.py")
    print(line_num)
    print(exist)

    print(search_manager.history["search_query"])


def test_search_callable_1():
    repo_path = "~/.orcar/astropy__astropy/"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    callable_name = "_scale_back_ascii"
    code_snippet = search_manager.search_callable(
        query_name=callable_name,
        file_path="astropy/io/fits/fitsrec.py",
    )

    print(code_snippet)

    res = search_manager.get_query_from_history(
        action="search_callable", input="astropy/io/fits/fitsrec.py::" + callable_name
    )
    print(res)


def test_search_callable_2():
    repo_path = "~/.orcar/django__django"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    callable_name = "to_python"
    code_snippet = search_manager.search_callable(
        query_name=callable_name,
        file_path="django/forms/models.py",
    )
    print(code_snippet)

    code_snippet = search_manager.search_callable(
        query_name=callable_name,
    )

    print(code_snippet)

    callable_name = "ModelChoiceField"
    code_snippet = search_manager.search_callable(
        query_name=callable_name,
    )

    print(code_snippet)

    search_class = search_manager.search_class(
        class_name="ASCIIUsernameValidator",
        file_path="django/contrib/auth/validators.py",
    )
    print(search_class)


def test_editor_get_bug_code():
    repo_path = "~/.orcar/django__django/"
    expand_repo_path = os.path.expanduser(repo_path)
    editor = Editor(repo_path=expand_repo_path)
    bug_code = editor._get_bug_code(
        "Command", "django/core/management/commands/sqlmigrate.py"
    )
    bug_code_handle = editor._get_bug_code(
        "handle", "django/core/management/commands/sqlmigrate.py"
    )
    print(bug_code)
    print(bug_code_handle)


def test_matplotlib_axesgrid():
    repo_path = "~/.orcar/matplotlib__matplotlib/"
    expand_repo_path = os.path.expanduser(repo_path)
    graph_builder = RepoGraph(
        repo_path=expand_repo_path, save_log=True, log_path="log", build_kg=True
    )
    node = graph_builder.dfs_search_callable_def("AxesGrid")
    if node:
        print(
            f"Snapshot of class AxesGrid: \
            \n {node}"
        )
    else:
        print("Class snapshot not found")


def test_inverted_index():
    repo_path = "~/.orcar/matplotlib__matplotlib/"
    expand_repo_path = os.path.expanduser(repo_path)
    graph_builder = RepoGraph(
        repo_path=expand_repo_path, save_log=True, log_path="log", build_kg=True
    )
    inverted_index = graph_builder.inverted_index
    print(inverted_index.search("AxesGrid"))


def test_disambuiguate():
    repo_path = "~/.orcar/matplotlib__matplotlib/"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    res = search_manager.search_class("AxesGrid")
    print(res)

    res = search_manager.search_class(
        class_name="AxesGrid", file_path="lib/mpl_toolkits/axes_grid1/axes_grid.py"
    )
    print(res)

    res = search_manager.search_class(
        class_name="AxesGrid", file_path="lib/mpl_toolkits/axisartist/axes_grid.py"
    )
    print(res)


def test_duplicate_search_history():
    repo_path = "~/.orcar/django__django"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    # first search using the search_callable method
    callable_name = "ModelChoiceField"
    search_manager.search_callable(
        query_name=callable_name,
        file_path="django/forms/models.py",
    )
    history_res = search_manager.get_frame_from_history(
        action="search_class", input="django/forms/models.py::" + callable_name
    )
    print(history_res)
    history_res = search_manager.get_frame_from_history(
        action="search_callable", input="django/forms/models.py::" + callable_name
    )
    print(history_res)
    # second search using the search_class method
    search_manager.search_class("ModelChoiceField")
    history_res = search_manager.get_frame_from_history(
        action="search_class", input="ModelChoiceField"
    )
    print(history_res)


def test_disambiguous_methods_1():
    repo_path = "~/.orcar/django__django"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    callable_name = "to_python"
    search_manager.search_callable(
        query_name=callable_name,
        file_path="django/forms/models.py",
    )
    history_res = search_manager.get_frame_from_history(
        action="search_callable", input="django/forms/models.py::" + callable_name
    )
    is_disambiguous = history_res["query_type"] == "disambiguation"
    print(f"Is disambiguous: {is_disambiguous}")
    # use _get_disambiguous_methods
    disambiguous_methods, methods_code = search_manager._get_disambiguous_methods(
        method_name=callable_name, class_name=None, file_path="django/forms/models.py"
    )
    print(disambiguous_methods)
    print(methods_code)


def test_disambiguous_methods_2():
    repo_path = "~/.orcar/matplotlib__matplotlib"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    callable_name = "AxesGrid"
    search_manager.search_callable(
        query_name=callable_name,
    )
    history_res = search_manager.get_frame_from_history(
        action="search_callable", input=callable_name
    )
    is_disambiguous = history_res["query_type"] == "disambiguation"
    print(f"Is disambiguous: {is_disambiguous}")
    # use _get_disambiguous_methods
    disambiguous_methods, methods_code = search_manager._get_disambiguous_methods(
        method_name=callable_name
    )
    print(disambiguous_methods)
    print(methods_code)


def test_disambiguous_methods_3():
    repo_path = "~/.orcar/matplotlib__matplotlib"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    class_name = "ImageGrid"
    code_snippet = search_manager.search_method_in_class(
        class_name=class_name,
        method_name="_init_locators",
    )
    print(code_snippet)

    code_snippet = search_manager.search_callable(
        query_name="_init_locators",
    )
    print(code_snippet)
    disambiguous_methods, methods_code = search_manager._get_disambiguous_methods(
        method_name="_init_locators"
    )
    print(disambiguous_methods)

    disambiguous_methods, methods_code = search_manager._get_disambiguous_methods(
        method_name="_init_locators",
        file_path="lib/mpl_toolkits/axes_grid1/axes_grid.py",
    )
    print(disambiguous_methods)


def test_file_contents_1():
    repo_path = "~/.orcar/django__django"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    file_contents = search_manager.search_file_contents(
        "models.py", directory_path="django/forms"
    )
    print(file_contents)

    file_contents = search_manager.search_file_contents(
        "validators.py", directory_path="django/contrib/auth"
    )
    print(file_contents)


def test_file_contents_2():
    repo_path = "~/.orcar/django__django"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)

    file_contents = search_manager.search_file_contents("sqlmigrate.py")
    print(file_contents)

    history_res = search_manager.get_frame_from_history(
        action="search_file_contents", input="sqlmigrate.py"
    )
    assert history_res["query_type"] == "file"

    file_contents = search_manager.search_file_contents("boundfield.py")
    print(file_contents)

    history_res = search_manager.get_frame_from_history(
        action="search_file_contents", input="boundfield.py"
    )
    assert history_res["is_skeleton"]

    disambiguation = search_manager.search_file_contents("query.py")
    print(disambiguation)

    history_res = search_manager.get_frame_from_history(
        action="search_file_contents", input="query.py"
    )

    assert history_res["query_type"] == "disambiguation"


def test_analyze_bug_locs_with_file():
    repo_path = "~/.orcar/django__django"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    bug_locs = search_manager._get_bug_location(
        "Command", "django/core/management/commands/sqlmigrate.py"
    )
    print(bug_locs)
    bug_locs = search_manager._get_bug_location("to_python", "django/forms/models.py")
    print(bug_locs)
    bug_locs = search_manager._get_bug_location(
        "ModelChoiceField", "django/forms/models.py"
    )
    print(bug_locs)
    bug_locs = search_manager._get_bug_location(
        "serializer_factory", "django/db/migrations/serializer.py"
    )
    print(bug_locs)


def test_analyze_bug_locs():
    repo_path = "~/.orcar/django__django"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    bug_locs = search_manager._get_bug_location(
        "to_python",
    )
    print(bug_locs)
    bug_locs = search_manager._get_bug_location(
        "ModelChoiceField",
    )
    print(bug_locs)
    bug_locs = search_manager._get_bug_location(
        "serializer_factory",
    )
    print(bug_locs)


def test_build_graph_sympy():
    repo_path = "~/.orcar/sympy__sympy"
    expand_repo_path = os.path.expanduser(repo_path)
    graph_builder = RepoGraph(
        repo_path=expand_repo_path, save_log=True, log_path="log", build_kg=True
    )
    # try to search function "add" in the graph
    node = graph_builder.dfs_search_callable_def("resolvent_coeff_lambdas")
    if node:
        print(f"Snapshot of callable resolvent_coeff_lambdas: \n {node}")
    else:
        print("Callable snapshot not found")


def test_retrieve_dependency():
    repo_path = "~/.orcar/django__django"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    dep_locs = search_manager._get_dependency(
        "django/contrib/admin/utils.py::display_for_value"
    )
    print(dep_locs)


def test_retrieve_dependency_2():
    repo_path = "~/.orcar/pytest-dev__pytest"
    expand_repo_path = os.path.expanduser(repo_path)
    search_manager = SearchManager(repo_path=expand_repo_path)
    # {
    #         "file_path": "src/_pytest/pytesters.py",
    #         "class_name": "Pytester",
    #         "method_name": "spawn"
    #     }
    dep_locs = search_manager._get_dependency(
        "src/_pytest/pytester.py::Pytester::spawn"
    )
    print(dep_locs)

def test_file_tree():
    repo_path = "../"
    repo_graph = RepoGraph(repo_path=repo_path, save_log=True, log_path="log", build_kg=True)
    file_tree = repo_graph.get_file_tree()
    print(file_tree)


if __name__ == "__main__":
    # Example usage
    # test_build_graph()
    # test_search_manager()
    # test_exact_search()
    # test_local_build_graph()
    # test_env_build_graph()
    # test_fitsrec()
    # test_fitsrec_source_code()
    # test_search_callable_1()
    # test_search_callable_2()
    # print_search_priority()
    # test_editor_get_bug_code()
    # test_matplotlib_axesgrid()
    # test_inverted_index()
    # test_disambuiguate()
    # test_duplicate_search_history()
    # test_disambiguous_methods_1()
    # test_disambiguous_methods_2()
    # test_disambiguous_methods_3()
    # test_file_contents_1()
    # test_file_contents_2()
    # test_analyze_bug_locs_with_file()
    # test_analyze_bug_locs()
    # test_build_graph_sympy()
    # test_retrieve_dependency()
    # test_retrieve_dependency_2()
    test_file_tree()
