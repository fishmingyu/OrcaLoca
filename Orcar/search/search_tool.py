import networkx as nx
from .build_graph import RepoGraph, Loc
from typing import Tuple
from collections.abc import MutableMapping
import os
import ast

class SearchManager:
    
    def __init__(self, repo_path: str):
        self.history = []
        self.repo_path = repo_path
        self._setup_graph()

    def _setup_graph(self):
        graph_builder = RepoGraph(repo_path=self.repo_path)
        self.kg = graph_builder


    def get_search_functions(self) -> list:
        """Return a list of search functions."""
        return [
            self.search_callable,
            self.search_func,
            self.search_class_skeleton,
            self.search_method_in_class,
            self.search_file_skeleton,
        ]

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
        
    def _search_source_code(self, file_path: str, source_code: str) -> str:
        """Search the source code in the file.

        Args:
            file_path (str): The file path to search.
            source_code (str): The source code to search.

        Returns:
            str: The related function/class code snippet. 
                If not found, return the error message.
        """
        with open(file_path, "r") as f:
            file_content = f.read()
        
        tree = ast.parse(file_content)

            # Traverse the AST to find all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno
                end_line = node.end_lineno
                # check if the source_code is in the function body
                func_body = self.get_code_snippet(file_path, start_line, end_line)
                if source_code in func_body:
                    return func_body
        return f"Cannot find the context of {source_code} in {file_path}"
    
    def _get_file_skeleton(self, file_name: str) -> str | None:
        """Get the skeleton of the file, including class and function definitions.

        Args:
            file_name (str): The file name to get the skeleton.

        Returns:
            str | None: The skeleton of the file or None if not found.
        """
        return self.kg.dfs_search_file_skeleton(file_name)

    def _search_callable_kg(self, callable: str) -> Loc:
        """Search the callable in the knowledge graph.

        Args:
            callable (str): The function or class name to search.
            it can be a class or a function name (including methods).

        Returns:
            Loc: The location of the function/class definition.
        """
        return self.kg.dfs_search_callable_def(callable)
    
    def _get_class_skeleton(self, class_name: str) -> str | None:
        """Search the class in the knowledge graph.

        Args:
            class_name (str): The class name to search.

        Returns:
            str | None: The skeleton of the class or None if not found.
        """
        return self.kg.get_class_snapshot(class_name)
    
    def _search_func_kg(self, func_name: str) -> Loc:
        """Search the function in the knowledge graph.

        Args:
            func_name (str): The function name to search.

        Returns:
            Loc: The location of the function definition.
        """
        return self.kg.dfs_search_func_def(func_name)
    
    def _search_method_in_class_kg(self, class_name: str, method_name: str) -> Loc:
        """Search the method in the knowledge graph.

        Args:
            class_name (str): The class name to search.
            method_name (str): The method name to search. 
                It should be the method name without the class name.
        Returns:
            Loc: The location of the method definition.
        """
        return self.kg.dfs_search_method_in_class(class_name, method_name)


    #################
    # Interface methods
    #################

    def search_file_skeleton(self, file_name: str) -> str:
        """API to search the file skeleton
            If you want to see the structure of the file, including class and function signatures.
            Be sure to call search_class_skeleton and search_func to get detailed information.

        Args:
            file_name (str): The file name to search. Usage: search_file_contents("example.py")
            Do not include the path, only the file name.

        Returns:
            str: The skeleton of the file.
        """
        res = self._get_file_skeleton(file_name)
        if res is None:
            return f"Cannot find the file skeleton of {file_name}"
        return res

    def search_callable(self, callable: str) -> Tuple[str, str]:
        """API to search the callable in given repo. Only if you can't make sure if it's a class or a function.

        Args:
            callable (str): The function or class name to search.

        Returns:
            Tuple[str, str]: The file path and the code_snippet of the callable definition.
        """
        loc = self._search_callable_kg(callable)
        if loc is None:
            return ("", f"Cannot find the definition of callable:{callable}")
        # loc.file_path is relative to the repo_path
        joined_path = os.path.join(self.repo_path, loc.file_name)
        return (loc.file_name, self.get_code_snippet(joined_path, loc.start_line, loc.end_line))
    
    def search_class_skeleton(self, class_name: str) -> str:
        """API to search the class skeleton in given repo.

        Args:
            class_name (str): The class name to search.

        Returns:
            str: The skeleton snapshot of the class. Including methods within the class.
            Please call search_method_in_class to get detailed information of the method.
            If the methods don't have docstrings, please make sure use search_method_in_class to get the method signature.
        """
        snapshot = self._get_class_skeleton(class_name)
        if snapshot is None:
            return f"Cannot find the class:{class_name}"
        return snapshot
    
    def search_func(self, func_name: str) -> Tuple[str, str]:
        """API to search the standalone function in given repo.
        NEVER use this API to search the method in the class. Use search_method_in_class instead.

        Args:
            func_name (str): The function name to search.

        Returns:
            Tuple[str, str]: The file path and the code_snippet of the function definition.
        """
        loc = self._search_func_kg(func_name)
        if loc is None:
            return ("", f"Cannot find the definition of function:{func_name}")
        joined_path = os.path.join(self.repo_path, loc.file_name)
        return (loc.file_name, self.get_code_snippet(joined_path, loc.start_line, loc.end_line))
    
    def search_method_in_class(self, class_name: str, method_name: str) -> Tuple[str, str]:
        """API to search the method in the class in given repo.
        
        Args:
            class_name (str): The class name to search.
            method_name (str): The method name within the class.

            Returns:
                Tuple[str, str]: The file path and the code_snippet of the method definition.
        """
        loc = self._search_method_in_class_kg(class_name, method_name)
        if loc is None:
            return ("", f"Cannot find the definition of method:{method_name} in class:{class_name}")
        joined_path = os.path.join(self.repo_path, loc.file_name)
        return (loc.file_name, self.get_code_snippet(joined_path, loc.start_line, loc.end_line))

    def search_source_code(self, file_path: str, source_code: str) -> str:
        """API to search the source code in the file.

        Args:
            file_path (str): The file path to search.
            source_code (str): The source code to search.

        Returns:
            str: The related function/method code snippet. 
                If not found, return the error message.
        """
        return self._search_source_code(file_path, source_code)