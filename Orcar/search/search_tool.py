import ast
import os
import re
from typing import List, Tuple

import pandas as pd

from .build_graph import Loc, LocInfo, RepoGraph


class SearchManager:
    def __init__(self, repo_path) -> None:
        self.history = pd.DataFrame(
            columns=[
                "search_action",
                "search_input",
                "search_query",
                "search_content",
                "query_type",
                "file_path",
                "is_skeleton",
            ]
        )
        self.repo_path = repo_path
        self._setup_graph()

    def _setup_graph(self):
        graph_builder = RepoGraph(repo_path=self.repo_path)
        self.kg = graph_builder
        self.inverted_index = graph_builder.inverted_index

    def get_search_functions(self) -> list:
        """Return a list of search functions."""
        return [
            self.search_file_skeleton,
            self.search_source_code,
            self.fuzzy_search,
            self.exact_search,
        ]

    def get_frame_from_history(self, action: str, input: str) -> pd.DataFrame:
        # Get the search result from the history using the search_action and search_input
        result = self.history[
            (self.history["search_action"] == action)
            & (self.history["search_input"] == input)
        ]
        return result

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

    def _get_exact_loc(self, query: str) -> LocInfo | None:
        """Get the exact location of the query in the knowledge graph.

        Args:
            query (str): The query to search.

        Returns:
            LocInfo | None: The location of the query.
        """
        return self.kg.get_query(query)

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
            Tuple[Loc, str] | None: The location of the file and the file skeleton.
        """
        return self.kg.dfs_search_file_skeleton(file_name)

    def _search_callable_kg(self, callable: str) -> LocInfo:
        """Search the callable in the knowledge graph.

        Args:
            callable (str): The function or class name to search.
            it can be a class or a function name (including methods).

        Returns:
            LocInfo: The location of the callable.
        """
        return self.kg.dfs_search_callable_def(callable)

    def _dfs_get_class(self, class_name: str) -> Tuple[Loc, str] | None:
        """Search the class in the knowledge graph.

        Args:
            class_name (str): The class name to search.

        Returns:
            Tuple[Loc, str] | None: The location of the class and the class skeleton.
        """
        return self.kg.dfs_get_class_snapshot(class_name)

    def _direct_get_class(self, class_node_name: str) -> str | None:
        """Search the class in the knowledge graph.

        Args:
            class_node_name (str): The class node name to search.

        Returns:
            str | None: The class snapshot.
        """
        return self.kg.direct_get_class_snapshot_from_node(class_node_name)

    def _get_class_methods(self, class_node_name: str) -> Tuple[List[str], List[str]]:
        """Return
        1. The list of method names in the class.
        2. The list of method code snippets in the class.
        """

        output_methods = []
        output_code_snippets = []
        list_loc = self.kg.get_class_methods(class_node_name)
        for loc in list_loc:
            file_name = loc.file_name
            node_name = loc.node_name
            joined_path = os.path.join(self.repo_path, file_name)
            content = self._get_code_snippet(joined_path, loc.start_line, loc.end_line)
            output_methods.append(node_name)
            output_code_snippets.append(content)
        return output_methods, output_code_snippets

    def _fuzzy_search_class(self, class_name: str) -> str:
        """API for fuzzy search the class in the given repo.
        It will only return one match. (Other matches will be ignored)

        Args:
            class_name (str): The class name to search.

        Returns:
            str: The file path and the class content. If the content exceeds 100 lines, we will use class skeleton.
        """
        loc, snapshot = self._dfs_get_class(class_name)
        # we don't check whether loc is global variable, since fuzzy search will help to disambiguate
        if loc is None:
            return f"Cannot find the class {class_name}"
        start_line = loc.start_line
        end_line = loc.end_line
        if end_line - start_line <= 100:
            joined_path = os.path.join(self.repo_path, loc.file_name)
            content = self._get_code_snippet(joined_path, start_line, end_line)
            new_row = {
                "search_action": "fuzzy_search",
                "search_input": class_name,
                "search_query": loc.node_name,
                "search_content": content,
                "query_type": "class",
                "file_path": loc.file_name,
                "is_skeleton": False,
            }
            self.history = pd.concat(
                [self.history, pd.DataFrame([new_row])], ignore_index=True
            )
            return f"""File Path: {loc.file_name} \nQuery Type: class \nClass Content: \n{content}"""

        new_row = {
            "search_action": "fuzzy_search",
            "search_input": class_name,
            "search_query": loc.node_name,
            "search_content": snapshot,
            "query_type": "class",
            "file_path": loc.file_name,
            "is_skeleton": True,
        }
        self.history = pd.concat(
            [self.history, pd.DataFrame([new_row])], ignore_index=True
        )

        return f"""File Path: {loc.file_name} \nQuery Type: class \nClass Skeleton: \n{snapshot}"""

    #################
    # Interface methods
    #################

    def search_file_skeleton(self, file_name: str) -> str:
        """API to search the file skeleton
            If you want to see the structure of the file, including class and function signatures.
            Be sure to call search_class and search_func to get detailed information in the file.

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
            "query_type": "file",
            "file_path": loc.file_name,
            "is_skeleton": True,
        }
        self.history = pd.concat(
            [self.history, pd.DataFrame([new_row])], ignore_index=True
        )
        return f"""File Path: {loc.file_name} \nFile Skeleton: \n{res}"""

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
            "search_input": source_code,
            "search_query": source_code,
            "search_content": content,
            "query_type": "source_code",
            "file_path": file_path,
            "is_skeleton": False,
        }
        self.history = pd.concat(
            [self.history, pd.DataFrame([new_row])], ignore_index=True
        )
        return f"""File Path: {file_path} \nCode Snippet: \n{content}"""

    def fuzzy_search(self, query: str) -> str:
        """API to search the query with fuzzy search.
            Use this API when you are not sure about the query type, query's file path.
            Notice you should only put the query name, not including any file path.
            E.g. fuzzy_search("ModelChoiceField")

        Args:
            query (str): The query to search.

        Returns:
            str:
            1. If we find multiple matches for a query, return the disambiguation message. The message contains the list of possible matches, where
            each match contains the file path, containing class (if exists). You need to use exact_search to get the detailed information.
            2. If we find only one match, return the file path, query type, and the code snippet. For class we have special treatment.
                If the class content is less than 100 lines, we return the class content. Otherwise, we return the class skeleton.
            3. If we cannot find the query, return the error message.

        """
        # first check the inverted_index, the inverted_index does not contain single value key
        # if the query is a single value key, we directly search in the knowledge graph
        if query in self.inverted_index.index:
            locs = self.inverted_index.search(query)
            # len(locs) is always greater than 1
            res = "Multiple matches found. Please use exact_search to get the detailed information.\n"
            for loc in locs:
                res += f"Possible Location {locs.index(loc)+1}:\n"
                res += f"File Path: {loc.file_path}\n"
                if loc.class_name:
                    res += f"Containing Class: {loc.class_name}\n"
                res += "\n"
            # add <Disambiguation>res</Disambiguation>
            ret_string = f"<Disambiguation>\n{res}</Disambiguation>"
            new_row = {
                "search_action": "fuzzy_search",
                "search_input": query,
                "search_query": query,
                "search_content": res,
                "query_type": "disambiguation",
                "file_path": "",
                "is_skeleton": False,
            }
            self.history = pd.concat(
                [self.history, pd.DataFrame([new_row])], ignore_index=True
            )
            return ret_string
        # if the query is not in the inverted_index, we search in the knowledge graph
        locinfo: LocInfo = self._search_callable_kg(query)
        if locinfo is None:
            return f"Cannot find the definition of {query}"
        loc = locinfo.loc
        type = locinfo.type

        joined_path = os.path.join(self.repo_path, loc.file_name)
        # if type is class, we use the class snapshot
        if type == "class":
            return self._fuzzy_search_class(query)

        content = self._get_code_snippet(joined_path, loc.start_line, loc.end_line)
        new_row = {
            "search_action": "fuzzy_search",
            "search_input": query,
            "search_query": loc.node_name,
            "search_content": content,
            "query_type": type,
            "file_path": loc.file_name,
            "is_skeleton": False,
        }
        self.history = pd.concat(
            [self.history, pd.DataFrame([new_row])], ignore_index=True
        )

        return f"""File Path: {loc.file_name} \nQuery Type: {type} \nCode Snippet: \n{content}"""

    def exact_search(
        self, query: str, file_path: str, containing_class: str | None = None
    ) -> str:
        """API to search the query with exact search.
            Use this API when you know the query's file path.
            If you know the query is a method, please provide the containing class name.
            E.g. exact_search("ModelChoiceField", "django/forms/models.py")
                 exact_search("to_python", "django/forms/models.py", "ModelChoiceField")
                 Adding the containing class name to avoid ambiguity.
                 However, if the query is a class, you MUST leave the containing class as None.

        Args:
            query (str): The query to search.
            file_path (str): The file path to search.
            containing_class (str): The containing class name. If the query is a method, provide the containing class name.
            Otherwise, leave it as None.

        Returns:
            str: The file path, and the code snippet.
        """
        # we use the query node format
        if containing_class is None:
            node_name = f"{file_path}::{query}"
        else:
            node_name = f"{file_path}::{containing_class}::{query}"
        loc_info = self._get_exact_loc(node_name)
        if loc_info is None:
            # sometimes the query is a class but the containing class is not None (LLM may misbehave)
            # fall back to use node_name = file_path::query
            loc_info = self._get_exact_loc(f"{file_path}::{query}")
            if loc_info is None:  # still cannot find the query
                return f"Cannot find the definition of {query} in {file_path}"
        type = loc_info.type
        loc = loc_info.loc
        node_name = loc.node_name

        joined_path = os.path.join(self.repo_path, loc.file_name)
        content = self._get_code_snippet(joined_path, loc.start_line, loc.end_line)

        if type == "method":
            search_input = f"{file_path}::{containing_class}::{query}"
        else:
            search_input = f"{file_path}::{query}"

        if type == "class":
            # if the type is class, we use the class snapshot
            snapshot = self._direct_get_class(node_name)
            start_line = loc.start_line
            end_line = loc.end_line
            if end_line - start_line > 100:  # use class skeleton
                new_row = {
                    "search_action": "exact_search",
                    "search_input": search_input,
                    "search_query": node_name,
                    "search_content": snapshot,
                    "query_type": type,
                    "file_path": loc.file_name,
                    "is_skeleton": True,
                }
                self.history = pd.concat(
                    [self.history, pd.DataFrame([new_row])], ignore_index=True
                )
                return f"""File Path: {loc.file_name} \nQuery Type: {type} \nClass Skeleton: \n{snapshot}"""

        new_row = {
            "search_action": "exact_search",
            "search_input": search_input,
            "search_query": node_name,
            "search_content": content,
            "query_type": type,
            "file_path": loc.file_name,
            "is_skeleton": False,
        }
        self.history = pd.concat(
            [self.history, pd.DataFrame([new_row])], ignore_index=True
        )

        return f"""File Path: {loc.file_name} \nQuery Type: {type} \nCode Snippet: \n{content}"""
