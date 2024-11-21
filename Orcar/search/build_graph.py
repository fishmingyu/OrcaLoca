import ast
import json
import os
from collections import namedtuple
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..environment.benchmark import BenchmarkEnv, get_repo_dir
from .inverted_index import InvertedIndex

Loc = namedtuple("Loc", ["file_name", "node_name", "start_line", "end_line"])
Snapshot = namedtuple("snapshot", ["docstring", "signature"])


class LocInfo:
    def __init__(self, loc: Loc, type: str):
        if not isinstance(loc, Loc):
            raise TypeError(f"loc must be of type Loc, not {type(loc)}")
        if not isinstance(type, str):
            raise TypeError(f"type must be a string, not {type(type)}")

        self.loc = loc
        self.type = type

    def __repr__(self):
        return f"LocInfo(loc={self.loc}, type={self.type})"


exclude_patterns = [
    "doc",  # Discovered this issue in 'pytest-dev__pytest'
    "requests/packages",  # Workaround for issue in 'psf__requests'
    "tests/regrtest_data",
    "tests/input",
    "tests/functional",  # Workaround for issue in 'pylint-dev__pylint'
    "tests/roots",
    "sphinx/templates/latex",  # Workaround for issue in 'sphinx-doc__sphinx'
    "tests/test_runner_apps/tagged/",
    "django/conf/app_template/",  # Workaround for issue in 'django__django'
]


# we use knowledge graph for faster retrieval of information
class RepoGraph:
    def __init__(
        self,
        swe_env: BenchmarkEnv = None,
        repo_path: str = None,
        save_log: bool = False,
        log_path: str = None,
        build_kg: bool = True,
    ):
        """
        swe_env: BenchmarkEnv
            The environment to build the graph (Caution: we don't recommend using this option,
            since it's extremely slow due to low rate of serial communication)
        repo_path: str
            The path to the repository to build the graph
        save_log: bool
            Whether to save the log
        log_path: str
            The path to save the log
        build_kg: bool
            Whether to build the knowledge graph
        """
        self.graph = nx.DiGraph()
        # Check that at least one of swe_env or repo_path is provided
        if swe_env is None and repo_path is None:
            raise ValueError("Either 'swe_env' or 'repo_path' must be provided.")
        # check swe_env is an instance of BenchmarkEnv
        if not isinstance(swe_env, BenchmarkEnv) and swe_env is not None:
            raise ValueError("swe_env must be an instance of BenchmarkEnv")
        if swe_env is not None:
            self.swe_env = swe_env
        elif repo_path is not None:
            self.swe_env = None
            self.repo_path = repo_path  # Path to the repository (absolute path)

        self.inverted_index = InvertedIndex()

        self.save_log = save_log
        self.log_path = log_path  # Name of the output log directory
        self.function_definitions = (
            {}
        )  # Map to store function definitions by their qualified name
        if build_kg:
            if self.swe_env is not None:
                self.repo_path = "/" + get_repo_dir(self.swe_env.ds.iloc[0]["repo"])
                self.build_whole_graph_from_env(self.repo_path, swe_env)
            else:
                self.build_whole_graph(repo_path)

    @staticmethod
    def extract_prefix(func_name):
        """
        Given a function name, extract the prefix of the function name
        """
        parts = func_name.split("::")
        if (
            len(parts) == 1
        ):  # no prefix, meaning it's a file or directory (not a function)
            return None
        return "::".join(parts[:-1])

    @staticmethod
    def extract_file_name(query):
        """
        Given a query, extract the file name from the query
        """
        path = query.split("::")[0]
        # then we get a path
        parts = path.split("/")
        return parts[-1]

    @staticmethod
    def check_class_prefix(func_name, prefix):
        """
        Given a function name and a prefix, check if the function name has the class prefix
        """
        # class prefix is the second last part of the function name
        parts = func_name.split("::")
        if (
            len(parts) == 1
        ):  # no prefix, meaning it's a file or directory (not a function)
            return False
        return parts[-2] == prefix

    # check node name exists in the graph
    def check_node_exists(self, node_name) -> bool:
        return node_name in self.graph.nodes

    # distance between two nodes, A, B
    def get_hops_between_nodes(self, A, B) -> int:
        hop_number = 0
        # A large constant to represent no path
        NO_PATH = 999999

        # Try A to B in the original direction
        try:
            hop_number_A_to_B = nx.shortest_path_length(self.graph, source=A, target=B)
        except nx.NetworkXNoPath:
            hop_number_A_to_B = NO_PATH  # No path found in this direction

        # Try B to A in the reversed direction
        try:
            hop_number_B_to_A = nx.shortest_path_length(self.graph, source=B, target=A)
        except nx.NetworkXNoPath:
            hop_number_B_to_A = NO_PATH  # No path found in this direction

        # Determine the smaller hop number if any path exists
        if hop_number_A_to_B == NO_PATH and hop_number_B_to_A == NO_PATH:
            # no path exists
            hop_number = NO_PATH
        else:
            hop_number = min(hop_number_A_to_B, hop_number_B_to_A)

        return hop_number

    def add_node(self, node_name, node_type, signature=None, docstring=None, loc=None):
        """Add a node to the graph with a type and its corresponding layer.
        node_type: directory, file, class, function, method
        signature: function signature
        docstring: function docstring
        loc: location of the node in the file (file_name, start_line, end_line)
        """
        layer = self._get_layer(node_type)
        self.graph.add_node(
            node_name,
            type=node_type,
            signature=signature,
            docstring=docstring,
            layer=layer,
            loc=loc,
        )

    def add_edge(self, from_node, to_node, edge_type):
        """Add an edge to the graph representing a dependency.
        edge_type: contains, references
        """
        self.graph.add_edge(from_node, to_node, edge_type=edge_type)

    @property
    def root_node(self):
        # check node name with "."
        for node in self.graph.nodes:
            if node == ".":
                return node

    @property
    def nodes_num(self):
        return self.graph.number_of_nodes()

    # high level search for the callable function or class definition in the graph
    def dfs_search_callable_def(self, query) -> LocInfo | None:
        root = self.root_node
        stack = [root]
        visited = set()
        kg_query_name = query
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                current_prefix = self.extract_prefix(node)
                if current_prefix is not None:
                    kg_query_name = current_prefix + "::" + query
                if node == kg_query_name:
                    return LocInfo(
                        loc=self.graph.nodes[node]["loc"],
                        type=self.graph.nodes[node]["type"],
                    )
                stack.extend(self.graph.neighbors(node))
        return None

    # constrained search for method definition in the graph (method is a function inside a class)
    def dfs_search_method_in_class(self, class_name, method_name) -> Loc | None:
        root = self.root_node
        stack = [root]
        visited = set()
        kg_query_name = method_name
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                current_prefix = self.extract_prefix(node)
                if current_prefix is not None:
                    kg_query_name = current_prefix + "::" + method_name
                if node == kg_query_name:
                    prefix_true = self.check_class_prefix(kg_query_name, class_name)
                    if self.graph.nodes[node]["type"] == "method" and prefix_true:
                        return self.graph.nodes[node]["loc"]
                stack.extend(self.graph.neighbors(node))
        return None

    # dfs search for the methods in a class and its docstring
    def get_class_snapshot(self, class_name) -> Tuple[Loc, str] | None:
        root = self.root_node
        stack = [root]
        visited = set()
        methods = {}
        class_snapshot = Snapshot("", "")
        kg_query_name = class_name
        loc = None

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                current_prefix = self.extract_prefix(node)
                if current_prefix is not None:
                    kg_query_name = current_prefix + "::" + class_name
                if node == kg_query_name:
                    # this is class node
                    # get all neighbors of this node means all methods
                    for method in self.graph.neighbors(node):
                        method_name = method.split("::")[-1]
                        method_snapshot = Snapshot(
                            self.graph.nodes[method]["docstring"],
                            self.graph.nodes[method]["signature"],
                        )
                        methods[method_name] = method_snapshot
                    class_snapshot = Snapshot(
                        self.graph.nodes[node]["docstring"],
                        self.graph.nodes[node]["signature"],
                    )
                    loc = self.graph.nodes[node]["loc"]
                stack.extend(self.graph.neighbors(node))
        # setup the snapshot
        snapshot = ""
        if loc:
            snapshot += f"Class Signature: {class_snapshot.signature}\n"
            snapshot += f"Docstring: {class_snapshot.docstring}\n"
        else:
            return None
        for method_name, method_snapshot in methods.items():
            snapshot += f"\nMethod: {method_name}\n"
            snapshot += f"Method Signature: {method_snapshot.signature}\n"
            snapshot += f"Docstring: {method_snapshot.docstring}\n"

        return loc, snapshot

    def get_class_methods(self, class_name) -> List[Loc] | None:
        root = self.root_node
        stack = [root]
        visited = set()
        methods = []
        kg_query_name = class_name

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                current_prefix = self.extract_prefix(node)
                if current_prefix is not None:
                    kg_query_name = current_prefix + "::" + class_name
                if node == kg_query_name:
                    # this is class node
                    # get all neighbors of this node means all methods
                    for method in self.graph.neighbors(node):
                        if self.graph.nodes[method]["type"] == "method":
                            methods.append(self.graph.nodes[method]["loc"])
                stack.extend(self.graph.neighbors(node))
        return methods

    # dfs search for contents in a file
    def dfs_search_file_skeleton(self, file_name) -> Tuple[Loc, str] | None:
        root = self.root_node
        stack = [root]
        visited = set()
        contents = {}
        loc = None

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                extracted_file = self.extract_file_name(node)
                # first check if the node is a file
                type_of_node = self.graph.nodes[node]["type"]
                if type_of_node == "file":
                    if extracted_file == file_name:
                        # this is file node
                        # get all neighbors of this node means all methods
                        loc = self.graph.nodes[node]["loc"]
                        for childs in self.graph.neighbors(node):
                            child_name = childs.split("::")[-1]
                            # child could be class/function or global variable
                            if self.graph.nodes[childs]["type"] in [
                                "function",
                                "class",
                            ]:
                                snapshot = (
                                    self.graph.nodes[childs]["type"],
                                    self.graph.nodes[childs]["signature"],
                                    self.graph.nodes[childs]["docstring"],
                                )
                                contents[child_name] = snapshot
                            else:  # global variable
                                contents[child_name] = (
                                    self.graph.nodes[childs]["type"],
                                    None,
                                    None,
                                )
                # only add neighbors that are files or directories
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor]["type"] in ["file", "directory"]:
                        stack.append(neighbor)
        # setup the snapshot
        snapshot = ""
        if contents:
            for child_name, child_snapshot in contents.items():
                node_type, signature, docstring = child_snapshot
                snapshot += f"\n{node_type.capitalize()}: {child_name}\n"
                if signature:
                    snapshot += f"Signature: {signature}\n"
                if docstring:
                    snapshot += f"Docstring: {docstring}\n"
            return loc, snapshot
        return None

    # dfs search for query in a given file
    def dfs_search_query_in_file(self, file_path, query) -> LocInfo | None:
        root = self.root_node
        stack = [root]
        visited = set()

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                extracted_file_path = node.split("::")[0]
                # first check if the node is a file
                type_of_node = self.graph.nodes[node]["type"]
                if type_of_node == "file":
                    if extracted_file_path == file_path:
                        # this is file node
                        # get all neighbors of this node means all methods
                        for childs in self.graph.neighbors(node):
                            child_name = childs.split("::")[-1]
                            # check if the child name is the query
                            if child_name == query:
                                locinfo = LocInfo(
                                    loc=self.graph.nodes[childs]["loc"],
                                    type=self.graph.nodes[childs]["type"],
                                )
                                return locinfo
                            # if child is a class, then we need to check all the methods in the class
                            if self.graph.nodes[childs]["type"] == "class":
                                for method in self.graph.neighbors(childs):
                                    method_name = method.split("::")[-1]
                                    if method_name == query:
                                        locinfo = LocInfo(
                                            loc=self.graph.nodes[method]["loc"],
                                            type=self.graph.nodes[method]["type"],
                                        )
                                        return locinfo
                # only add neighbors that are files or directories
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor]["type"] in ["file", "directory"]:
                        stack.append(neighbor)
        return None

    # build the graph from the repository
    def build_attribute_from_repo(self, repo_path):
        """Build a graph from a repository."""
        for root, dirs, files in os.walk(repo_path):
            # Add a node for the directory
            dir_node_name = os.path.relpath(root, repo_path)
            # Skip directories that match any exclude pattern
            if any(dir_node_name.startswith(pattern) for pattern in exclude_patterns):
                continue
            loc = Loc(
                file_name=dir_node_name,
                node_name=dir_node_name,
                start_line=0,
                end_line=0,
            )
            self.add_node(dir_node_name, "directory", loc=loc)

            dirs[:] = [
                sub_dir
                for sub_dir in dirs
                if not any(
                    os.path.join(dir_node_name, sub_dir).startswith(pattern)
                    for pattern in exclude_patterns
                )
            ]

            # Process each subdirectory
            for sub_dir in dirs:
                sub_dir_path = os.path.join(root, sub_dir)
                sub_dir_node_name = os.path.relpath(sub_dir_path, repo_path)

                # Add a node for each subdirectory
                loc = Loc(
                    file_name=sub_dir_node_name,
                    node_name=sub_dir_node_name,
                    start_line=0,
                    end_line=0,
                )
                self.add_node(sub_dir_node_name, "directory", loc=loc)
                self.add_edge(dir_node_name, sub_dir_node_name, "contains")

            for file in files:
                # now only consider python files
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    file_node_name = os.path.relpath(file_path, repo_path)

                    # Skip files that match any exclude pattern
                    if any(
                        file_node_name.startswith(pattern)
                        for pattern in exclude_patterns
                    ):
                        continue

                    # Add a node for the file and link it to the directory
                    loc = Loc(
                        file_name=file_node_name,
                        node_name=file_node_name,
                        start_line=0,
                        end_line=0,
                    )
                    self.add_node(file_node_name, "file", loc=loc)
                    self.add_edge(dir_node_name, file_node_name, "contains")

                    # Build the graph for the file's content
                    self.build_attribute_from_file(file_path, file_node_name)

        return self.graph

    def build_attribute_from_env_repo(self, repo_path):
        """Build a graph from a repository in the environment."""
        for root, dirs, files in self.swe_env.walk(repo_path):
            # Add a node for the directory
            dir_node_name = os.path.relpath(root, repo_path)
            loc = Loc(
                file_name=dir_node_name,
                node_name=dir_node_name,
                start_line=0,
                end_line=0,
            )
            self.add_node(dir_node_name, "directory", loc=loc)

            # Skip directories that match any exclude pattern
            if any(dir_node_name.startswith(pattern) for pattern in exclude_patterns):
                continue

            dirs[:] = [
                sub_dir
                for sub_dir in dirs
                if not any(
                    os.path.join(dir_node_name, sub_dir).startswith(pattern)
                    for pattern in exclude_patterns
                )
            ]

            # Process each subdirectory
            for sub_dir in dirs:
                sub_dir_path = os.path.join(root, sub_dir)
                sub_dir_node_name = os.path.relpath(sub_dir_path, repo_path)

                # Add a node for each subdirectory
                loc = Loc(
                    file_name=sub_dir_node_name,
                    node_name=sub_dir_node_name,
                    start_line=0,
                    end_line=0,
                )
                self.add_node(sub_dir_node_name, "directory")
                self.add_edge(dir_node_name, sub_dir_node_name, "contains")

            for file in files:
                # now only consider python files
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    file_node_name = os.path.relpath(file_path, repo_path)

                    # Skip files that match any exclude pattern
                    if any(
                        file_node_name.startswith(pattern)
                        for pattern in exclude_patterns
                    ):
                        continue

                    # Add a node for the file and link it to the directory
                    loc = Loc(
                        file_name=file_node_name,
                        node_name=file_node_name,
                        start_line=0,
                        end_line=0,
                    )
                    self.add_node(file_node_name, "file", loc=loc)
                    self.add_edge(dir_node_name, file_node_name, "contains")

                    # Build the graph for the file's content
                    self.build_attribute_from_env_file(file_path, file_node_name)

        return self.graph

    def build_attribute_from_file(self, file_path, file_node_name):
        """Build a graph from a single file."""
        with open(file_path, "r") as file:
            tree = ast.parse(file.read())

        # Build the graph for this file's content
        visitor = AstVistor(
            self, file_node_name, self.function_definitions, self.inverted_index
        )
        visitor.visit(tree)

    def build_attribute_from_env_file(self, file_path, file_node_name):
        """Build a graph from a single file in the environment."""
        file = self.swe_env.read_text_file(file_path)
        tree = ast.parse(file)

        # Build the graph for this file's content
        visitor = AstVistor(
            self, file_node_name, self.function_definitions, self.inverted_index
        )
        visitor.visit(tree)

    def build_references(self, repo_path):
        """Build reference edges between functions in the graph."""
        knowledge_graph = self.graph
        self.reference_builder = ReferenceBuilder(
            knowledge_graph, self.function_definitions
        )
        self.reference_builder.build_references(repo_path)

    def build_references_from_env(self, repo_path, swe_env: BenchmarkEnv):
        """Build reference edges between functions in the graph."""
        knowledge_graph = self.graph
        self.reference_builder = ReferenceBuilder(
            knowledge_graph, self.function_definitions, swe_env
        )
        self.reference_builder.build_references_from_env(repo_path)

    def build_whole_graph(self, repo_path):
        """Build the whole graph from a repository."""
        self.build_attribute_from_repo(repo_path)
        self.build_references(repo_path)
        if self.save_log:
            if self.log_path is None:
                self.log_path = "log"
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.dump_graph()
            if self.nodes_num < 100:  # only save the graph if it's small
                self.save_graph()
            # self.save_graph()

    def build_whole_graph_from_env(self, repo_path, swe_env: BenchmarkEnv):
        """Build the whole graph from a repository in the environment."""
        self.build_attribute_from_env_repo(repo_path)
        self.build_references_from_env(repo_path, swe_env)
        if self.save_log:
            if self.log_path is None:
                self.log_path = "log"
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.dump_graph()
            if self.nodes_num < 100:
                self.save_graph()

    def dump_graph(self):
        """Dump the graph as a dictionary."""
        data = nx.node_link_data(self.graph)
        log_path = self.log_path
        filename = os.path.join(log_path, "repo_graph.json")
        with open(filename, "w") as file:
            json.dump(data, file)

    def save_graph(self):
        """Save the graph as a JPG file using a tree-like layout."""
        # Use pygraphviz layout
        log_path = self.log_path
        filename = os.path.join(log_path, "repo_graph.jpg")
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")

        node_colors = [
            self._get_node_color(data["type"])
            for _, data in self.graph.nodes(data=True)
        ]
        labels = {
            node: f"{os.path.basename(node)}"
            for node, data in self.graph.nodes(data=True)
        }
        plt.figure(figsize=(15, 15))

        # Draw the graph with arrows
        nx.draw(
            self.graph,
            pos,
            labels=labels,
            with_labels=True,
            node_size=3000,
            node_color=node_colors,
            font_size=10,
            font_weight="bold",
            arrows=True,
            connectionstyle="arc3,rad=0.1",
            edge_color=[
                self._get_edge_color(edge_type)
                for _, _, edge_type in self.graph.edges(data="edge_type")
            ],
            style=[
                self._get_edge_style(edge_type)
                for _, _, edge_type in self.graph.edges(data="edge_type")
            ],
            width=[
                self._get_edge_width(edge_type)
                for _, _, edge_type in self.graph.edges(data="edge_type")
            ],
            arrowstyle="-|>",
            arrowsize=20,
        )
        # Add legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                linestyle="solid",
                label="Contains Relationship",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                lw=1,
                linestyle="dotted",
                label="References Relationship",
            ),
            Patch(facecolor="lightgreen", edgecolor="black", label="Directory"),
            Patch(facecolor="lightblue", edgecolor="black", label="File"),
            Patch(facecolor="orange", edgecolor="black", label="Class"),
            Patch(facecolor="pink", edgecolor="black", label="Function/Method"),
            Patch(facecolor="yellow", edgecolor="black", label="Global/Class Variable"),
        ]
        plt.legend(handles=legend_elements, loc="upper right", fontsize="large")

        plt.savefig(filename, format="jpg")
        plt.close()

    def _get_layer(self, node_type):
        """Determine the layer based on the node type."""
        if node_type == "directory":
            return 0
        elif node_type == "file":
            return 1
        elif node_type == "class":
            return 2
        elif node_type in ["function", "method"]:
            return 3
        else:  # global_variable, class_variable
            return 4

    def _get_node_color(self, node_type):
        """Return color based on the node type."""
        if node_type == "directory":
            return "lightgreen"
        elif node_type == "file":
            return "lightblue"
        elif node_type == "class":
            return "orange"
        elif node_type in ["function", "method"]:
            return "pink"
        else:  # global_variable, class_variable
            return "yellow"

    def _get_edge_style(self, edge_type):
        return "solid" if edge_type == "contains" else "dotted"

    def _get_edge_color(self, edge_type):
        return "black" if edge_type == "contains" else "gray"

    def _get_edge_width(self, edge_type):
        return (
            2 if edge_type == "contains" else 1
        )  # Thicker width for 'contains', thinner for 'references'


class AstVistor(ast.NodeVisitor):
    def __init__(
        self, graph_builder, file_node_name, function_definitions, inverted_index
    ):
        self.graph_builder = graph_builder
        self.function_definitions = function_definitions
        self.current_class = None
        self.current_function = None
        self.current_file = file_node_name
        self.inverted_index = inverted_index

    def visit_FunctionDef(self, node):
        function_name = node.name
        args = [arg.arg for arg in node.args.args]
        signature = f"{function_name}({', '.join(args)})"
        docstring = ast.get_docstring(node)

        node_name = (
            f"{self.current_file}::{function_name}"
            if self.current_class is None
            else f"{self.current_class}::{function_name}"
        )
        self.function_definitions[function_name] = node_name
        node_type = "function" if self.current_class is None else "method"
        node_loc = Loc(
            file_name=self.current_file,
            node_name=node_name,
            start_line=node.lineno,
            end_line=node.end_lineno,
        )

        self.graph_builder.add_node(
            node_name, node_type, signature, docstring, node_loc
        )

        # Link function or method to the file node or class node
        parent_node = self.current_class if self.current_class else self.current_file
        self.graph_builder.add_edge(parent_node, node_name, "contains")

        # Set the current function context
        previous_function = self.current_function
        self.current_function = node_name

        self.generic_visit(node)

        # Restore the previous function context
        self.current_function = previous_function

    def visit_ClassDef(self, node):
        class_name = node.name
        docstring = ast.get_docstring(node)

        node_name = f"{self.current_file}::{class_name}"
        node_loc = Loc(
            file_name=self.current_file,
            node_name=node_name,
            start_line=node.lineno,
            end_line=node.end_lineno,
        )
        self.graph_builder.add_node(node_name, "class", class_name, docstring, node_loc)

        # Link class to the file node
        self.graph_builder.add_edge(self.current_file, node_name, "contains")

        previous_class = self.current_class
        self.current_class = node_name  # use parent class as the current class

        self.generic_visit(node)
        # save the previous class
        self.current_class = previous_class

    def visit_Assign(self, node):
        # Only handle global assignments
        if self.current_class is None and self.current_function is None:
            # It's a global variable
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    node_name = f"{self.current_file}::{var_name}"
                    node_loc = Loc(
                        file_name=self.current_file,
                        node_name=node_name,
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                    )
                    self.graph_builder.add_node(
                        node_name, "global_variable", var_name, None, node_loc
                    )
                    self.graph_builder.add_edge(
                        self.current_file, node_name, "contains"
                    )

        self.generic_visit(node)


# frist collect all the function definitions and then build the references
class ReferenceBuilder:
    def __init__(self, graph, function_definitions, swe_env: BenchmarkEnv = None):
        self.graph = graph  # The graph produced by RepoGraph
        self.function_definitions = function_definitions
        self.swe_env = swe_env

    def build_references(self, repo_path):
        """Build reference edges between functions in the graph."""
        for root, _, files in os.walk(repo_path):
            # Relative path of the current directory
            dir_node_name = os.path.relpath(root, repo_path)
            # Skip directories that match any exclude pattern
            if any(dir_node_name.startswith(pattern) for pattern in exclude_patterns):
                continue
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    rel_file_path = os.path.relpath(file_path, repo_path)
                    # Skip files that match any exclude pattern
                    if any(
                        rel_file_path.startswith(pattern)
                        for pattern in exclude_patterns
                    ):
                        continue
                    with open(file_path, "r") as file:
                        tree = ast.parse(file.read())
                        self._visit_tree(tree, rel_file_path)

    def build_references_from_env(self, repo_path):
        """Build reference edges between functions in the graph."""
        for root, _, files in self.swe_env.walk(repo_path):
            # Relative path of the current directory
            dir_node_name = os.path.relpath(root, repo_path)
            # Skip directories that match any exclude pattern
            if any(dir_node_name.startswith(pattern) for pattern in exclude_patterns):
                continue
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    rel_file_path = os.path.relpath(file_path, repo_path)
                    # Skip files that match any exclude pattern
                    if any(
                        rel_file_path.startswith(pattern)
                        for pattern in exclude_patterns
                    ):
                        continue
                    with self.swe_env.read_text_file(file_path, "r") as file:
                        tree = ast.parse(file)
                        self._visit_tree(tree, rel_file_path)

    def _visit_tree(self, tree, file_path):
        """Visit the tree to find function definitions and references."""
        visitor = FunctionReferenceVisitor(
            self.graph, self.function_definitions, file_path
        )
        visitor.visit(tree)


class FunctionReferenceVisitor(ast.NodeVisitor):
    def __init__(self, graph, function_definitions, file_path):
        self.graph = graph
        self.function_definitions = function_definitions
        self.current_file = file_path
        self.current_class = None
        self.current_function = None

    def visit_FunctionDef(self, node):
        function_name = node.name
        full_function_name = (
            f"{self.current_file}::{function_name}"
            if self.current_class is None
            else f"{self.current_file}::{self.current_class}::{function_name}"
        )
        self.current_function = full_function_name

        self.generic_visit(node)
        self.current_function = None  # Reset after visiting

    def visit_ClassDef(self, node):
        class_name = node.name
        self.current_class = class_name
        self.generic_visit(node)
        self.current_class = None  # Reset after visiting the class

    def visit_Call(self, node):
        """Capture function calls and add them as edges in the graph."""
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
            if function_name in self.function_definitions:
                caller = self.current_function
                callee = self.function_definitions[function_name]
                if caller and callee:
                    self.graph.add_edge(caller, callee, edge_type="references")
        self.generic_visit(node)
