from Orcar.search import RepoGraph, SearchManager

def test_build_graph():
    repo_path = "../../django"
    graph_builder = RepoGraph(repo_path, save_log=True, log_path="log", build_kg=True)
    # try to search function "add" in the graph
    kg_graph = graph_builder.graph
    root = graph_builder.root_node
    node = graph_builder.dfs_search_function_def("ModelChoiceField")
    if node:
        print(f"Found the function definition at {node}")
    else:
        print("Function definition not found")


def test_search_manager():
    repo_path = "../../django"
    search_manager = SearchManager(repo_path)
    # try to search function "add" in the graph
    file_path, code_snippet = search_manager.search_func("ModelChoiceField")
    print(code_snippet)

if __name__ == "__main__":
    # Example usage
    # test_build_graph()
    test_search_manager()