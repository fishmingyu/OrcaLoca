import ast
import os
import re
from typing import Tuple

from .build_graph import Loc, LocInfo, RepoGraph


class SearchManager:
    def __init__(self, repo_path) -> None:
        self.history = []
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

    @property
    def search_tool_priority(self) -> str:
        """Return the search tool priority."""
        # The search tool priority is the order of the search functions in the list below
        list = [
            self.search_file_skeleton,
            self.search_class_skeleton,
            self.search_method_in_class,
            self.search_callable,
            self.search_source_code,
        ]
        # convert the list to a description string, use the name of the function
        return f"""
        Please follow the search tool priority below:
        1. {list[0].__name__}
        2. {list[1].__name__}
        3. {list[2].__name__}: use this after you have already tried {list[1].__name__},
        4. {list[3].__name__}: use this when you can't make sure the query is a method,
        5. {list[4].__name__}: only use source code search when you can't find the query in the file.
        """

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
        return f"""
        File Path: {loc.file_name} \n
        Method Code Snippet: \n
        {self._get_code_snippet(joined_path, loc.start_line, loc.end_line)}
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
        return f"""
        File Path: {file_path} \n
        Code Snippet: \n
        {self._search_source_code(file_path, source_code)}
        """

    def search_callable(self, query: str, **kwargs) -> str:
        """API to search the callable definition in the file.
        The query can be a function, class, method or global variable. Don't use this API to search source code in the file.

        Args:
            query (str): The query to search. It can be a function, class, method or global variable.

        KwArgs: (Optional)
            file_path (str): The file path to search. If not provided, search in the whole repo.

        Returns:
            str: The file path and the code snippet of the query. If query type is class, return the class skeleton.
        """
        if "file_path" in kwargs:
            file_path = kwargs["file_path"]
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

        return f"""
        File Path: {loc.file_name} \n
        Code Snippet or Skeleton (if class): \n
        {self._get_code_snippet(joined_path, loc.start_line, loc.end_line)}
        """
