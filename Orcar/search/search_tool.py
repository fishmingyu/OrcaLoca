import networkx as nx
from .build_graph import RepoGraph, Loc
from typing import Tuple
from collections.abc import MutableMapping
import os

class SearchManager:
    
    def __init__(self, repo_path: str):
        self.history = []
        self.repo_path = repo_path
        self._setup_graph()

    def _setup_graph(self):
        graph_builder = RepoGraph(self.repo_path)
        graph_builder.build_whole_graph(self.repo_path)
        self.kg = graph_builder

    def get_code_snippet(self, file_path: str, start: int, end: int) -> str:
        """Get the code snippet in the range in the file, without line numbers.

        Args:
            file_path (str): Full path to the file.
            start (int): Start line number. (1-based)
            end (int): End line number. (1-based)

        Returns:
            str: The code snippet.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()
            return "".join(lines[start-1:end])
        
    def _search_func_kg(self, func_name: str) -> Loc:
        """Search the function in the knowledge graph.

        Args:
            func_name (str): The function name to search.

        Returns:
            Loc: The location of the function definition.
        """
        return self.kg.dfs_search_function_def(func_name)


    #################
    # Interface methods
    #################

    def search_func(self, func_name: str) -> Tuple[str, str]:
        """Search the function in the knowledge graph.

        Args:
            func_name (str): The function name to search.

        Returns:
            Tuple[str, str]: The file path and the code_snippet of the function definition.
        """
        loc = self._search_func_kg(func_name)
        # loc.file_path is relative to the repo_path
        joined_path = os.path.join(self.repo_path, loc.file_name)
        return (loc.file_name, self.get_code_snippet(joined_path, loc.start_line, loc.end_line))
