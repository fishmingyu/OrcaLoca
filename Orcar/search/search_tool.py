import ast
import os
import re
from typing import List, Tuple

import pandas as pd

from .build_graph import Loc, LocInfo, RepoGraph
from .inverted_index import IndexValue, InvertedIndex


def check_class_method_unique(
    class_name: str, method_name: str, index: InvertedIndex
) -> bool:
    """Check if the class method is unique in the inverted index.

    Args:
        class_name (str): The class name to check.
        method_name (str): The method name to check.
        index (InvertedIndex): The inverted index.

    Returns:
        bool: True if the class method is unique, False otherwise.
    """
    key = method_name
    if key not in index.index:  # if the method is not in the index, it is unique
        return True
    locs = index.search(key)
    hit = 0
    for loc in locs:
        # if more than one class contains the method, it is not unique (they may in different files)
        if loc.class_name == class_name:
            hit += 1
    if hit > 1:
        return False
    return True


def check_callable_unique_in_file(
    callable: str, file_path: str, index: InvertedIndex
) -> bool:
    """Check if the callable is unique in the file.

    Args:
        callable (str): The callable name to check.
        file_path (str): The file path to check.
        index (InvertedIndex): The inverted index.

    Returns:
        bool: True if the callable is unique in the file, False otherwise.
    """
    key = callable
    if key not in index.index:  # if the callable is not in the index, it is unique
        return True
    locs = index.search(key)
    hit = 0
    for loc in locs:
        # if more than one callable exists in the file, it is not unique
        if loc.file_path == file_path:
            hit += 1
    if hit > 1:
        return False
    return True


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
            self.search_class,
            self.search_method_in_class,
            self.search_callable,
        ]

    def get_frame_from_history(self, action: str, input: str) -> pd.DataFrame | None:
        # Get the search result from the history using the search_action and search_input
        result = self.history[
            (self.history["search_action"] == action)
            & (self.history["search_input"] == input)
        ]
        # return the first match if there are multiple matches
        # since they are the same
        return result.iloc[0] if not result.empty else None

    def get_query_from_history(self, action: str, input: str) -> str | None:
        # Get the search_query from the history using the search_action and search_input
        result = self.history[
            (self.history["search_action"] == action)
            & (self.history["search_input"] == input)
        ]["search_query"]

        if not result.empty:
            # return the first match if there are multiple matches
            query = result.iloc[0]
        else:
            query = None

        return query

    def get_node_existence(self, query: str) -> bool:
        # Check if the query exists in the knowledge graph
        if query is None:
            return False
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

    def _dfs_get_file_skeleton(self, file_name: str) -> Tuple[Loc, str] | None:
        """Get the skeleton of the file, including class and function definitions.

        Args:
            file_name (str): The file name to get the skeleton.

        Returns:
            Tuple[Loc, str] | None: The location of the file and the file skeleton.
        """
        return self.kg.dfs_search_file_skeleton(file_name)

    def _direct_get_file_skeleton(self, file_node_name: str) -> str | None:
        """Get the skeleton of the file, including class and function definitions.

        Args:
            file_node_name (str): The file node name to get the skeleton.

        Returns:
            str | None: The file skeleton.
        """
        return self.kg.direct_get_file_skeleton_from_node(file_node_name)

    def _search_callable_kg(self, callable: str) -> LocInfo | None:
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

    def _get_disambiguous_classes(
        self,
        class_name: str,
    ) -> List[str]:
        """Get the disambiguous classes

        Args:
            class_name (str): The class name to search.

        Returns:
            List[str]: The list of corresponding file paths.
        """
        output_files = []

        key = class_name
        if key not in self.inverted_index.index:
            return output_files
        index_values = self.inverted_index.search(key)
        for iv in index_values:
            output_files.append(iv.file_path)
        return output_files

    def _get_disambiguous_query_type(self, query: str) -> str | None:
        """
        Here we sample the first locinfo to get the query type, since we believe usually the query type is unique even if the query is not unique.
        """
        if query not in self.inverted_index.index:
            return None
        index_values = self.inverted_index.search(query)
        for iv in index_values:
            return iv.type

    def _get_disambiguous_methods(
        self, method_name: str, class_name: str = None, file_path: str = None
    ) -> Tuple[List[str], List[str]]:
        """Get the disambiguous methods in the file.

        Args:
            method_name (str): The method name to search.
            class_name (str): The containing class name. If the method is in a class, provide the class name.
            file_path (str): The file path to search. If you could make sure the file path, please provide it to avoid ambiguity.
            Leave it as None if you are not sure about the file path.

        Returns:
            Tuple[List[str], List[str]]: The list of method names and the list of method code snippets.
        """
        output_methods = []
        output_code_snippets = []

        def util_loc_output(iv: IndexValue) -> Tuple[str, str] | None:
            containing_class = iv.class_name
            iv_file_path = iv.file_path
            if containing_class is not None:
                # if the method is in a class, the node_name is file_path::class_name::method_name
                node_name = f"{iv_file_path}::{containing_class}::{method_name}"
            else:
                # if the method is not in a class, the node_name is file_path::method_name
                node_name = f"{iv_file_path}::{method_name}"
            locinfo = self._get_exact_loc(node_name)
            if locinfo is not None:
                loc = locinfo.loc
                joined_path = os.path.join(self.repo_path, loc.file_name)
                content = self._get_code_snippet(
                    joined_path, loc.start_line, loc.end_line
                )
                return loc.node_name, content
            return None

        if file_path is not None and class_name is not None:
            # use direct search; usually this wouldn't happen in disambiguation
            node_name = f"{file_path}::{class_name}::{method_name}"
            locinfo = self._get_exact_loc(node_name)
            if locinfo is not None:
                loc = locinfo.loc
                joined_path = os.path.join(self.repo_path, loc.file_name)
                content = self._get_code_snippet(
                    joined_path, loc.start_line, loc.end_line
                )
                output_methods.append(loc.node_name)
                output_code_snippets.append(content)
            return output_methods, output_code_snippets
        elif file_path is not None and class_name is None:
            # use inverted index, filter by file_path
            key = method_name
            if key not in self.inverted_index.index:
                return output_methods, output_code_snippets
            index_values = self.inverted_index.search(key)
            for iv in index_values:
                if iv.file_path == file_path:
                    # if util_loc_output returns None, we skip this iv
                    res = util_loc_output(iv)
                    if res is not None:
                        node_name, content = res
                        output_methods.append(node_name)
                        output_code_snippets.append(content)
            return output_methods, output_code_snippets
        elif file_path is None and class_name is not None:
            # use inverted index, filter by class_name
            key = method_name
            if key not in self.inverted_index.index:
                return output_methods, output_code_snippets
            index_values = self.inverted_index.search(key)
            for iv in index_values:
                if iv.class_name == class_name:
                    res = util_loc_output(iv)
                    if res is not None:
                        node_name, content = res
                        output_methods.append(node_name)
                        output_code_snippets.append(content)
            return output_methods, output_code_snippets
        else:
            # use inverted index, no filter
            key = method_name
            if key not in self.inverted_index.index:
                return output_methods, output_code_snippets
            index_values = self.inverted_index.search(key)
            for iv in index_values:
                res = util_loc_output(iv)
                if res is not None:
                    node_name, content = res
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

    def search_file_contents(
        self, file_name: str, directory_path: str | None = None
    ) -> str:
        """API to search the file skeleton
            If you want to see the structure of the file, including class and function signatures.
            Be sure to call search_class and search_method_in_class to get the detailed information.

        Args:
            file_name (str): The file name to search. Usage: search_file_contents("example.py"). Do not include the path, only the file name.
            directory_path (str): The directory path to search. Usage: search_file_contents("example.py", "path/to/directory")

        Returns:
           str: If file contents exceed 200 lines, we will return the file skeleton, a string that contains the file path and the file skeleton.
                Otherwise, we will return the file path and the file contents.
        """
        if directory_path is not None:
            file_node = f"{directory_path}/{file_name}"
            locinfo = self._get_exact_loc(file_node)
            if locinfo is None:
                return f"Cannot find the file {file_name} in {directory_path}"
            loc = locinfo.loc
            start_line = loc.start_line
            end_line = loc.end_line
            if end_line - start_line <= 200:
                content = self._get_code_snippet(
                    os.path.join(self.repo_path, loc.file_name), start_line, end_line
                )
                new_row = {
                    "search_action": "search_file_contents",
                    "search_input": file_name,
                    "search_query": loc.node_name,
                    "search_content": content,
                    "query_type": "file",
                    "file_path": loc.file_name,
                    "is_skeleton": False,
                }
                self.history = pd.concat(
                    [self.history, pd.DataFrame([new_row])], ignore_index=True
                )
                return f"""File Path: {loc.file_name} \nFile Content: \n{content}"""
            else:
                snapshot = self._direct_get_file_skeleton(loc.node_name)
                new_row = {
                    "search_action": "search_file_contents",
                    "search_input": file_name,
                    "search_query": loc.node_name,
                    "search_content": snapshot,
                    "query_type": "file",
                    "file_path": loc.file_name,
                    "is_skeleton": True,
                }
                self.history = pd.concat(
                    [self.history, pd.DataFrame([new_row])], ignore_index=True
                )
                return f"""File Path: {loc.file_name} \nFile Skeleton: \n{snapshot}"""
        else:
            # first check inverted_index, the inverted_index does not contain single value key
            # if the query is a single value key, we directly search in the knowledge graph
            if file_name in self.inverted_index.index:
                locs = self.inverted_index.search(file_name)
                # len(locs) is always greater than 1
                res = f"Multiple matches found for file {file_name}. \n"
                for loc in locs:
                    res += f"Possible Location {locs.index(loc)+1}:\n"
                    res += f"File Path: {loc.file_path}\n"
                    res += "\n"
                # add <Disambiguation>res</Disambiguation>
                ret_string = f"<Disambiguation>\n{res}</Disambiguation>"
                new_row = {
                    "search_action": "search_file_contents",
                    "search_input": file_name,
                    "search_query": file_name,
                    "search_content": res,
                    "query_type": "disambiguation",
                    "file_path": "",
                    "is_skeleton": False,
                }
                self.history = pd.concat(
                    [self.history, pd.DataFrame([new_row])], ignore_index=True
                )
                return ret_string
            loc, res = self._dfs_get_file_skeleton(file_name)
            if loc is None:
                return f"Cannot find the file {file_name}"
            start_line = loc.start_line
            end_line = loc.end_line
            if end_line - start_line <= 200:
                content = self._get_code_snippet(
                    os.path.join(self.repo_path, loc.file_name), start_line, end_line
                )
                new_row = {
                    "search_action": "search_file_contents",
                    "search_input": file_name,
                    "search_query": loc.node_name,
                    "search_content": content,
                    "query_type": "file",
                    "file_path": loc.file_name,
                    "is_skeleton": False,
                }
                self.history = pd.concat(
                    [self.history, pd.DataFrame([new_row])], ignore_index=True
                )
                return f"""File Path: {loc.file_name} \nFile Content: \n{content}"""
            else:
                new_row = {
                    "search_action": "search_file_contents",
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

    def search_class(self, class_name: str, file_path: str = None) -> str:
        """API to search the class in the given repo.

        Args:
            class_name (str): The class name to search.
            file_path (str): The file path to search. If you could make sure the file path, please provide it to avoid ambiguity.
            Leave it as None if you are not sure about the file path.
            Usage: search_class("ModelChoiceField") or search_class("ModelChoiceField", "django/forms/models.py")

        Returns:
            str: The file path and the class content. If the content exceeds 100 lines, we will use class skeleton.
            If not found, return the error message. If multiple classes are found, return the disambiguation message.
            Please call search_method_in_class to get detailed information of the method after skeleton search.
            If the methods don't have docstrings, please make sure use search_method_in_class to get the method signature.
        """
        if file_path is not None:
            # use direct search
            node_name = f"{file_path}::{class_name}"
            search_input = node_name
            locinfo = self._get_exact_loc(node_name)
            if locinfo is None:
                return f"Cannot find the class {class_name} in {file_path}"
            loc = locinfo.loc
            start_line = loc.start_line
            end_line = loc.end_line
            if end_line - start_line <= 100:
                joined_path = os.path.join(self.repo_path, loc.file_name)
                content = self._get_code_snippet(joined_path, start_line, end_line)
                new_row = {
                    "search_action": "search_class",
                    "search_input": search_input,
                    "search_query": loc.node_name,
                    "search_content": content,
                    "query_type": "class",
                    "file_path": loc.file_name,
                    "is_skeleton": False,
                }
                self.history = pd.concat(
                    [self.history, pd.DataFrame([new_row])], ignore_index=True
                )
                return f"""File Path: {loc.file_name} \nClass Content: \n{content}"""
            else:  # use class skeleton
                snapshot = self._direct_get_class(loc.node_name)
                new_row = {
                    "search_action": "search_class",
                    "search_input": search_input,
                    "search_query": loc.node_name,
                    "search_content": snapshot,
                    "query_type": "class",
                    "file_path": loc.file_name,
                    "is_skeleton": True,
                }
                self.history = pd.concat(
                    [self.history, pd.DataFrame([new_row])], ignore_index=True
                )
                return f"""File Path: {loc.file_name} \nClass Skeleton: \n{snapshot}"""
        else:
            # first check the inverted_index, the inverted_index does not contain single value key
            # if the query is a single value key, we directly search in the knowledge graph
            # usually class name wouldn't be used as a method name, so we don't need to check the type
            if class_name in self.inverted_index.index:
                locs = self.inverted_index.search(class_name)
                # len(locs) is always greater than 1
                res = f"Multiple matched classes found about class: {class_name}. \n"
                for loc in locs:
                    res += f"Possible Location {locs.index(loc)+1}:\n"
                    res += f"File Path: {loc.file_path}\n"
                    res += "\n"
                # add <Disambiguation>res</Disambiguation>
                ret_string = f"<Disambiguation>\n{res}</Disambiguation>"
                new_row = {
                    "search_action": "search_class",
                    "search_input": class_name,
                    "search_query": class_name,
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
            loc, snapshot = self._dfs_get_class(class_name)
            if loc is None:
                return f"Cannot find the class {class_name}"
            start_line = loc.start_line
            end_line = loc.end_line
            if end_line - start_line <= 100:
                joined_path = os.path.join(self.repo_path, loc.file_name)
                content = self._get_code_snippet(joined_path, start_line, end_line)
                new_row = {
                    "search_action": "search_class",
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
                return f"""File Path: {loc.file_name} \nClass Content: \n{content}"""
            else:  # use class skeleton
                new_row = {
                    "search_action": "search_class",
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
                return f"""File Path: {loc.file_name} \nClass Skeleton: \n{snapshot}"""

    def search_method_in_class(
        self, class_name: str, method_name: str, file_path: str = None
    ) -> str:
        """API to search the method of the class in the given repo.
        Don't try to use this API until you have already tried search_class to get the class info.

        Args:
            class_name (str): The class name to search.
            method_name (str): The method name within the class.
            file_path (str): The file path to search. If you could make sure the file path, please provide it to avoid ambiguity.
            Leave it as None if you are not sure about the file path.
            Usage: search_method_in_class("ModelChoiceField", "to_python") or search_method_in_class("ModelChoiceField", "to_python", "django/forms/models.py")

        Returns:
            str: The file path and the method code snippet. If not found, return the error message.
            If multiple methods are found, return the disambiguation message.
        """
        if file_path is not None:
            # use direct search
            node_name = f"{file_path}::{class_name}::{method_name}"
            search_input = node_name
            locinfo = self._get_exact_loc(node_name)
            if locinfo is None:
                return f"Cannot find the method {method_name} in {class_name} in {file_path}"
            loc = locinfo.loc
            start_line = loc.start_line
            end_line = loc.end_line
            joined_path = os.path.join(self.repo_path, loc.file_name)
            content = self._get_code_snippet(joined_path, start_line, end_line)
            new_row = {
                "search_action": "search_method_in_class",
                "search_input": search_input,
                "search_query": loc.node_name,
                "search_content": content,
                "query_type": "method",
                "file_path": loc.file_name,
                "is_skeleton": False,
            }
            self.history = pd.concat(
                [self.history, pd.DataFrame([new_row])], ignore_index=True
            )
            return f"""File Path: {loc.file_name} \nMethod Content: \n{content}"""
        else:
            # first check the inverted_index, the inverted_index does not contain single value key
            # if the query is a single value key, we directly search in the knowledge graph
            search_input = f"{class_name}::{method_name}"
            if not check_class_method_unique(
                class_name, method_name, self.inverted_index
            ):  # if the class method is not unique
                locs = self.inverted_index.search(method_name)
                # len(locs) is always greater than 1
                res = f"Multiple matched methods found about method: {method_name} in class: {class_name}. \n"
                new_index = 0
                for loc in locs:
                    # filter class_name matches
                    if loc.class_name is None:  # not a method
                        continue
                    loc_class_name = loc.class_name
                    if loc_class_name != class_name:
                        continue
                    res += f"Possible Location {new_index+1}:\n"
                    res += f"File Path: {loc.file_path}\n"
                    res += f"Containing Class: {loc.class_name}\n"
                    new_index += 1
                # add <Disambiguation>res</Disambiguation>
                ret_string = f"<Disambiguation>\n{res}</Disambiguation>"
                new_row = {
                    "search_action": "search_method_in_class",
                    "search_input": search_input,
                    "search_query": method_name,
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
            # however here we should first detect whether the class is unique
            # since the knowledge graph may lead to a wrong class
            if class_name in self.inverted_index.index:
                locs = self.inverted_index.search(class_name)
                # note that there only exist one class that have corresponding method, since we have checked the uniqueness
                file_path = None
                for loc in locs:
                    # iterate to find the file_path
                    file_path = loc.file_path
                    node_name = f"{file_path}::{class_name}::{method_name}"
                    locinfo = self._get_exact_loc(node_name)
                    if locinfo is not None:  # method is found
                        break
                if file_path is None:  # no method is found
                    return f"Cannot find the method {method_name} in {class_name}"
                else:
                    node_name = f"{file_path}::{class_name}::{method_name}"
                    locinfo = self._get_exact_loc(node_name)
                    if locinfo is None:
                        return f"Cannot find the method {method_name} in {class_name} in {file_path}"
                    loc = locinfo.loc
                    start_line = loc.start_line
                    end_line = loc.end_line
                    joined_path = os.path.join(self.repo_path, loc.file_name)
                    content = self._get_code_snippet(joined_path, start_line, end_line)
                    new_row = {
                        "search_action": "search_method_in_class",
                        "search_input": search_input,
                        "search_query": loc.node_name,
                        "search_content": content,
                        "query_type": "method",
                        "file_path": loc.file_name,
                        "is_skeleton": False,
                    }
                    self.history = pd.concat(
                        [self.history, pd.DataFrame([new_row])], ignore_index=True
                    )
                    return (
                        f"""File Path: {loc.file_name} \nMethod Content: \n{content}"""
                    )
            node_name = f"{class_name}::{method_name}"
            loc = self.kg.dfs_search_method_in_class(class_name, method_name)
            if loc is None:
                return f"Cannot find the method {method_name} in {class_name}"
            start_line = loc.start_line
            end_line = loc.end_line
            joined_path = os.path.join(self.repo_path, loc.file_name)
            content = self._get_code_snippet(joined_path, start_line, end_line)
            new_row = {
                "search_action": "search_method_in_class",
                "search_input": node_name,
                "search_query": loc.node_name,
                "search_content": content,
                "query_type": "method",
                "file_path": loc.file_name,
                "is_skeleton": False,
            }
            self.history = pd.concat(
                [self.history, pd.DataFrame([new_row])], ignore_index=True
            )
            return f"""File Path: {loc.file_name} \nMethod Content: \n{content}"""

    def search_callable(self, query_name: str, file_path: str = None) -> str:
        """API to search the callable definition in the given repo.
        If you are not sure about the query type, please use this API. The query can be a function, class, method or global variable.

        Args:
            query_name (str): The query to search. The format should be only the name.
            file_path (str): The file path to search. If you could make sure the file path, please provide it to avoid ambiguity.
            Leave it as None if you are not sure about the file path.
            Usage: search_callable("ModelChoiceField") or search_callable("ModelChoiceField", "django/forms/models.py")

        Returns:
            str: The file path and the code snippet. If not found, return the error message.
            If multiple matches are found, return the disambiguation message.
        """
        if file_path is not None:  # got file_path hint
            # two cases: 1. the query is a method 2. the query is a class
            # first check callable uniqueness in the file
            search_input = f"{file_path}::{query_name}"
            if not check_callable_unique_in_file(
                query_name, file_path, self.inverted_index
            ):
                locs = self.inverted_index.search(query_name)
                res = f"Multiple matched callables found about query {query_name} in {file_path}. \n"
                new_index = 0
                for loc in locs:
                    if loc.file_path != file_path:
                        continue
                    res += f"Possible Location {new_index+1}:\n"
                    res += f"File Path: {loc.file_path}\n"
                    new_index += 1
                    if loc.class_name:
                        res += f"Containing Class: {loc.class_name}\n"
                    res += "\n"
                # add <Disambiguation>res</Disambiguation>
                ret_string = f"<Disambiguation>\n{res}</Disambiguation>"
                new_row = {
                    "search_action": "search_callable",
                    "search_input": search_input,
                    "search_query": query_name,
                    "search_content": res,
                    "query_type": "disambiguation",
                    "file_path": file_path,
                    "is_skeleton": False,
                }
                self.history = pd.concat(
                    [self.history, pd.DataFrame([new_row])], ignore_index=True
                )
                return ret_string
            else:  # unique
                # use get_query_in_file
                locinfo = self._get_query_in_file(file_path, query_name)
                if locinfo is None:
                    return f"Cannot find the definition of {query_name} in {file_path}"
                loc = locinfo.loc
                type = locinfo.type
                if (
                    type == "class"
                ):  # if the type is class, we fallback to use search_class
                    # add a new row to the history
                    content = self.search_class(query_name, file_path)
                    is_skeleton = (loc.end_line - loc.start_line) > 100
                    new_row = {
                        "search_action": "search_callable",
                        "search_input": search_input,
                        "search_query": loc.node_name,
                        "search_content": content,
                        "query_type": type,
                        "file_path": loc.file_name,
                        "is_skeleton": is_skeleton,
                    }
                    self.history = pd.concat(
                        [self.history, pd.DataFrame([new_row])], ignore_index=True
                    )
                    # we could handle duplicate history in pd frame(since they would be the same)
                    return content
                # if not class, we use the code snippet
                joined_path = os.path.join(self.repo_path, loc.file_name)
                content = self._get_code_snippet(
                    joined_path, loc.start_line, loc.end_line
                )
                new_row = {
                    "search_action": "search_callable",
                    "search_input": search_input,
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

        else:  # no file_path hint
            # first check the inverted_index, the inverted_index does not contain single value key
            # if the query is a single value key, we directly search in the knowledge graph
            if query_name in self.inverted_index.index:
                locs = self.inverted_index.search(query_name)
                # len(locs) is always greater than 1
                res = f"Multiple matched callables found about query {query_name}. \n"
                new_index = 0
                for loc in locs:
                    res += f"Possible Location {new_index+1}:\n"
                    res += f"File Path: {loc.file_path}\n"
                    new_index += 1
                    if loc.class_name:
                        res += f"Containing Class: {loc.class_name}\n"
                    res += "\n"
                # add <Disambiguation>res</Disambiguation>
                ret_string = f"<Disambiguation>\n{res}</Disambiguation>"
                new_row = {
                    "search_action": "search_callable",
                    "search_input": query_name,
                    "search_query": query_name,
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
            locinfo = self._search_callable_kg(query_name)
            if locinfo is None:
                return f"Cannot find the definition of {query_name}"
            loc = locinfo.loc
            type = locinfo.type
            if type == "class":  # if the type is class, we fallback to use search_class
                # add a new row to the history
                content = self.search_class(query_name)
                is_skeleton = (loc.end_line - loc.start_line) > 100
                new_row = {
                    "search_action": "search_callable",
                    "search_input": query_name,
                    "search_query": loc.node_name,
                    "search_content": content,
                    "query_type": type,
                    "file_path": loc.file_name,
                    "is_skeleton": is_skeleton,
                }
                self.history = pd.concat(
                    [self.history, pd.DataFrame([new_row])], ignore_index=True
                )
                # we could handle duplicate history in pd frame(since they would be the same)
                return content
            # if not class, we use the code snippet
            joined_path = os.path.join(self.repo_path, loc.file_name)
            content = self._get_code_snippet(joined_path, loc.start_line, loc.end_line)
            new_row = {
                "search_action": "search_callable",
                "search_input": query_name,
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
