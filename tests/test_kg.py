from Orcar.search import RepoGraph, SearchManager

def test_build_graph():
    repo_path = "../../django"
    graph_builder = RepoGraph(repo_path, save_log=True, log_path="log", build_kg=True)
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
    search_manager = SearchManager(repo_path)
    # try to search function "to_python" in ModelChoiceField class
    file_path, code_snippet = search_manager.search_method_in_class("ModelChoiceField", "to_python")
    print(code_snippet)

def test_local_build_graph():
    repo_graph = "./test_repo"
    graph_builder = RepoGraph(repo_graph, save_log=True, log_path="log", build_kg=True)
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


if __name__ == "__main__":
    # Example usage
    test_build_graph()
    # test_search_manager()
    # test_local_build_graph()