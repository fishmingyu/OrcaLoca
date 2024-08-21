from Orcar.search import RepoGraph, SearchManager

def test_build_graph():
    repo_path = "./test_repo"
    graph_builder = RepoGraph(repo_path, save_jpg=True, out_file_name="repo_graph.jpg", build_kg=True)
    # try to search function "add" in the graph
    node = graph_builder.dfs_search_function_def("get_sum")
    if node:
        print(f"Found the function definition at {node}")
    else:
        print("Function definition not found")


def test_search_manager():
    repo_path = "./test_repo"
    search_manager = SearchManager(repo_path)
    # try to search function "add" in the graph
    file_path, code_snippet = search_manager.search_func("add")
    print(code_snippet)

if __name__ == "__main__":
    # Example usage
    test_build_graph()
    test_search_manager()