import ast
import os
import re
from typing import List, Tuple

import pandas as pd

from .build_graph import Loc, LocInfo, RepoGraph


class SearchManager:
    def __init__(self, repo_path) -> None:
        self.history = pd.DataFrame(
            columns=["search_action", "search_input", "search_query", "search_content"]
        )
        self.repo_path = repo_path
        self._setup_graph()

    def _setup_graph(self):
        graph_builder = RepoGraph(repo_path=self.repo_path)
        self.kg = graph_builder

    def get_search_functions(self) -> list:
        """Return a list of search functions."""
        return [
            self.search_class_skeleton,
            self.search_method_in_class,
            self.search_file_skeleton,
            self.search_callable,
            self.search_source_code,
        ]

    def get_query_from_history(self, action: str, input: str) -> str | None:
        # Get the search_query from the history using the search_action and search_input
        result = self.history[
            (self.history["search_action"] == action)
            & (self.history["search_input"] == input)
        ]["search_query"]

        if not result.empty:
            query = result.values[0]
        else:
            query = None

        return query

    def get_node_existance(self, query: str) -> bool:
        # Check if the query exists in the knowledge graph
        return self.kg.check_node_exists(query)

    def get_distance_between_queries(self, query1: str, query2: str) -> int:
        if query1 is None or query2 is None:
            return -1
        else:
            return self.kg.get_hops_between_nodes(query1, query2)

    def _get_code_snippet(self, file_path: str, start: int, end: int) -> str:
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
            return "".join(lines[start - 1 : end])

    def _get_query_in_file(self, file_path: str, query: str) -> LocInfo | None:
        """Get the query definition in the file. Search the query in KG.

        Args:
            file_path (str): The file path to search.
            query (str): The query to search. It can be a function, class, method or global variable.

        Returns:
            LocInfo | None: The locinfo of the query.
        """
        return self.kg.dfs_search_query_in_file(file_path, query)

    def _search_source_code(self, file_path: str, source_code: str) -> str:
        """Search the source code in the file.
        Do not use this method to search source code in the file.

        Args:
            file_path (str): The file path to search.
            source_code (str): The source code to search.

        Returns:
            str: The related function/class code snippet.
                If not found, return the error message.
        """
        abs_file_path = os.path.join(self.repo_path, file_path)
        with open(abs_file_path, "r") as f:
            file_content = f.read()

        def normalize_code(code: str) -> str:
            """Normalize whitespace in the code."""
            # Remove leading and trailing whitespace
            code = code.strip()
            # Replace all sequences of whitespace (space, tab, newline) with a single space
            code = re.sub(r"\s+", " ", code)
            return code

        source_code = normalize_code(source_code)
        tree = ast.parse(file_content)

        # Traverse the AST to find all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno
                end_line = node.end_lineno
                # check if the source_code is in the function body
                func_body = self._get_code_snippet(abs_file_path, start_line, end_line)
                normalized_func_body = normalize_code(func_body)
                if source_code in normalized_func_body:
                    return func_body
        return f"Cannot find the context of {source_code} in {file_path}"

    def _get_file_skeleton(self, file_name: str) -> Tuple[Loc, str] | None:
        """Get the skeleton of the file, including class and function definitions.

        Args:
            file_name (str): The file name to get the skeleton.

        Returns:
            str | None: The skeleton of the file or None if not found.
        """
        return self.kg.dfs_search_file_skeleton(file_name)

    def _search_callable_kg(self, callable: str) -> LocInfo:
        """Search the callable in the knowledge graph.

        Args:
            callable (str): The function or class name to search.
            it can be a class or a function name (including methods).

        Returns:
            Loc: The location of the function/class definition.
        """
        return self.kg.dfs_search_callable_def(callable)

    def _get_class_skeleton(self, class_name: str) -> Tuple[Loc, str] | None:
        """Search the class in the knowledge graph.

        Args:
            class_name (str): The class name to search.

        Returns:
            str | None: The skeleton of the class or None if not found.
        """
        return self.kg.get_class_snapshot(class_name)

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

    def _get_class_methods(self, class_name: str) -> Tuple[List[str], List[str]]:
        """Return
        1. The list of method names in the class.
        2. The list of method code snippets in the class.
        """

        output_methods = []
        output_code_snippets = []
        list_loc = self.kg.get_class_methods(class_name)
        for loc in list_loc:
            file_name = loc.file_name
            node_name = loc.node_name
            joined_path = os.path.join(self.repo_path, file_name)
            content = self._get_code_snippet(joined_path, loc.start_line, loc.end_line)
            output_methods.append(node_name)
            output_code_snippets.append(content)
        return output_methods, output_code_snippets

    #################
    # Interface methods
    #################

    def search_file_skeleton(self, file_name: str) -> str:
        """API to search the file skeleton
            If you want to see the structure of the file, including class and function signatures.
            Be sure to call search_class_skeleton and search_func to get detailed information in the file.

        Args:
            file_name (str): The file name to search. Usage: search_file_contents("example.py")
            Do not include the path, only the file name.

        Returns:
           str: A string that contains the file path and the file skeleton.
        """

        loc, res = self._get_file_skeleton(file_name)
        if loc is None:
            return f"Cannot find the file skeleton of {file_name}"

        new_row = {
            "search_action": "search_file_skeleton",
            "search_input": file_name,
            "search_query": loc.node_name,
            "search_content": res,
        }
        self.history = pd.concat(
            [self.history, pd.DataFrame([new_row])], ignore_index=True
        )
        return f"""
        File Path: {loc.file_name} \n
        File Skeleton: \n
        {res}
        """

    def search_class_skeleton(self, class_name: str) -> str:
        """API to search the class skeleton in given repo.

        Args:
            class_name (str): The class name to search.

        Returns:
            str: The file path and the class skeleton.
            Please call search_method_in_class to get detailed information of the method after skeleton search.
            If the methods don't have docstrings, please make sure use search_method_in_class to get the method signature.
        """
        loc, snapshot = self._get_class_skeleton(class_name)
        if loc is None:
            return f"Cannot find the class skeleton of {class_name}"
        new_row = {
            "search_action": "search_class_skeleton",
            "search_input": class_name,
            "search_query": loc.node_name,
            "search_content": snapshot,
        }
        self.history = pd.concat(
            [self.history, pd.DataFrame([new_row])], ignore_index=True
        )
        return f"""
        File Path: {loc.file_name} \n
        Class Skeleton: \n
        {snapshot}
        """

    def search_method_in_class(self, class_name: str, method_name: str) -> str:
        """API to search the method in the class in given repo.
        Don't try to use this API until you have already tried search_class_skeleton to get the class skeleton.
        If you know the class name and method name.

        Args:
            class_name (str): The class name to search.
            method_name (str): The method name within the class.

            Returns:
                str: The file path and the method code snippet.
                If not found, return the error message.
        """
        loc = self._search_method_in_class_kg(class_name, method_name)

        if loc is None:
            return (
                f"Cannot find the definition of method:{method_name} in class:{class_name}",
            )
        joined_path = os.path.join(self.repo_path, loc.file_name)
        content = self._get_code_snippet(joined_path, loc.start_line, loc.end_line)
        new_row = {
            "search_action": "search_method_in_class",
            "search_input": f"{class_name}::{method_name}",
            "search_query": loc.node_name,
            "search_content": content,
        }
        self.history = pd.concat(
            [self.history, pd.DataFrame([new_row])], ignore_index=True
        )
        return f"""
        File Path: {loc.file_name} \n
        Method Code Snippet: \n
        {content}
        """

    def search_source_code(self, file_path: str, source_code: str) -> str:
        """API to search the source code in the file. If you want to search the code snippet in the file.

        Args:
            file_path (str): The file path to search.
            source_code (str): The source code to search.

        Returns:
            str: The file path and the related function/class code snippet.
                If not found, return the error message.
        """
        content = self._search_source_code(file_path, source_code)
        new_row = {
            "search_action": "search_source_code",
            "search_input": file_path,
            "search_query": source_code,
            "search_content": content,
        }
        self.history = pd.concat(
            [self.history, pd.DataFrame([new_row])], ignore_index=True
        )
        return f"""
        File Path: {file_path} \n
        Code Snippet: \n
        {content}
        """

    def search_callable(self, query: str, file_path: str = None) -> str:
        """API to search the callable definition in the file.
        The query can be a function, class, method or global variable. Don't use this API to search source code in the file.

        Args:
            query (str): The query to search. It can be a function, class, method or global variable.
            The format should be only the name.
            E.g. search_callable("function_name")

            file_path (str): The file path to search. If not provided, search in the whole repo.
            Provide the file path if you know the file for better performance.
            This is optional. E.g. search_callable("function_name", "a/example.py")

        Returns:
            str: The file path and the code snippet of the query. If query type is class, return the class skeleton.
        """
        if file_path is not None:
            locinfo: LocInfo = self._get_query_in_file(file_path, query)
        else:
            locinfo: LocInfo = self._search_callable_kg(query)
        if locinfo is None:
            return f"Cannot find the definition of {query}"
        loc = locinfo.loc
        type = locinfo.type

        joined_path = os.path.join(self.repo_path, loc.file_name)
        # if type is class, we use the class snapshot
        if type == "class":
            return self.search_class_skeleton(query)

        content = self._get_code_snippet(joined_path, loc.start_line, loc.end_line)
        new_row = {
            "search_action": "search_callable",
            "search_input": query,
            "search_query": loc.node_name,
            "search_content": content,
        }
        self.history = pd.concat(
            [self.history, pd.DataFrame([new_row])], ignore_index=True
        )

        return f"""
        File Path: {loc.file_name} \n
        Code Snippet or Skeleton (if class): \n
        {content}
        """
