import os
import ast
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# we use knowledge graph for faster retrieval of information
class GraphBuilder:
    def __init__(self, repo_path, save_jpg=False, out_file_name=None):
        self.graph = nx.DiGraph()
        self.save_jpg = save_jpg
        self.repo_path = repo_path  # Path to the repository
        self.out_file_name = out_file_name # Name of the output file
        self.function_definitions = {}  # Map to store function definitions by their qualified name

    def add_node(self, node_name, node_type, signature=None, docstring=None, loc=None):
        """Add a node to the graph with a type and its corresponding layer.
        node_type: directory, file, class, function, method
        signature: function signature
        docstring: function docstring
        loc: location of the node in the file (file_name, start_line, end_line)
        """
        layer = self._get_layer(node_type)
        self.graph.add_node(node_name, type=node_type, signature=signature, docstring=docstring, layer=layer, loc=loc)

    def add_edge(self, from_node, to_node, edge_type):
        """Add an edge to the graph representing a dependency.
        edge_type: contains, references
        """
        self.graph.add_edge(from_node, to_node, edge_type=edge_type)

    def build_graph_from_repo(self, repo_path):
        """Build a graph from a repository."""
        for root, dirs, files in os.walk(repo_path):
            # Add a node for the directory
            dir_node_name = os.path.relpath(root, repo_path)
            self.add_node(dir_node_name, 'directory')

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    file_node_name = os.path.relpath(file_path, repo_path)

                    # Add a node for the file and link it to the directory
                    self.add_node(file_node_name, 'file')
                    self.add_edge(dir_node_name, file_node_name, 'contains')

                    # Build the graph for the file's content
                    self.build_graph_from_file(file_path, file_node_name)

        return self.graph

    def build_graph_from_file(self, file_path, file_node_name):
        """Build a graph from a single file."""
        with open(file_path, "r") as file:
            tree = ast.parse(file.read())

        # Build the graph for this file's content
        visitor = FunctionClassVisitor(self, file_node_name, self.function_definitions)
        visitor.visit(tree)

    def build_references(self, repo_path):
        """Build reference edges between functions in the graph."""
        knowledge_graph = self.graph
        self.reference_builder = ReferenceBuilder(knowledge_graph, graph_builder.function_definitions)
        self.reference_builder.build_references(repo_path)

    def build_whole_graph(self, repo_path):
        """Build the whole graph from a repository."""
        self.build_graph_from_repo(repo_path)
        self.build_references(repo_path)
        if self.save_jpg:
            self.save_graph(self.out_file_name)

    def save_graph(self, filename=None):
        """Save the graph as a JPG file using a tree-like layout."""
        # Use pygraphviz layout
        if filename is None:
            filename = self.out_file_name
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")

        node_colors = [self._get_node_color(data['type']) for _, data in self.graph.nodes(data=True)]
        labels = {
            node: f"{os.path.basename(node)}" for node, data in self.graph.nodes(data=True)
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
            edge_color=[self._get_edge_color(edge_type) for _, _, edge_type in self.graph.edges(data='edge_type')],
            style=[self._get_edge_style(edge_type) for _, _, edge_type in self.graph.edges(data='edge_type')],
            width=[self._get_edge_width(edge_type) for _, _, edge_type in self.graph.edges(data='edge_type')],
            arrowstyle='-|>',
            arrowsize=20
        )
        # Add legend
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, linestyle='solid', label='Contains Relationship'),
            Line2D([0], [0], color='black', lw=1, linestyle='dotted', label='References Relationship'),
            Patch(facecolor='lightgreen', edgecolor='black', label='Directory'),
            Patch(facecolor='lightblue', edgecolor='black', label='File'),
            Patch(facecolor='orange', edgecolor='black', label='Class'),
            Patch(facecolor='pink', edgecolor='black', label='Function/Method')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize='large')

        plt.savefig(filename, format="jpg")
        plt.close()

    def _get_layer(self, node_type):
        """Determine the layer based on the node type."""
        if node_type == 'directory':
            return 0
        elif node_type == 'file':
            return 1
        elif node_type == 'class':
            return 2
        else:  # function or method
            return 3

    def _get_node_color(self, node_type):
        """Return color based on the node type."""
        if node_type == 'directory':
            return 'lightgreen'
        elif node_type == 'file':
            return 'lightblue'
        elif node_type == 'class':
            return 'orange'
        else:  # function or method
            return 'pink'
        
    def _get_edge_style(self, edge_type):
        return 'solid' if edge_type == 'contains' else 'dotted'

    def _get_edge_color(self, edge_type):
        return 'black' if edge_type == 'contains' else 'gray'

    def _get_edge_width(self, edge_type):
        return 2 if edge_type == 'contains' else 1  # Thicker width for 'contains', thinner for 'references'


class FunctionClassVisitor(ast.NodeVisitor):
    def __init__(self, graph_builder, file_node_name, function_definitions):
        self.graph_builder = graph_builder
        self.function_definitions = function_definitions
        self.current_class = None
        self.current_file = file_node_name

    def visit_FunctionDef(self, node):
        function_name = node.name
        args = [arg.arg for arg in node.args.args]
        signature = f"{function_name}({', '.join(args)})"
        docstring = ast.get_docstring(node)
        
        node_name = f"{self.current_file}::{function_name}" if self.current_class is None else f"{self.current_class}::{function_name}"
        self.function_definitions[function_name] = node_name
        node_type = 'function' if self.current_class is None else 'method'
        node_loc = (self.current_file, node.lineno, node.end_lineno)
        
        self.graph_builder.add_node(node_name, node_type, signature, docstring, node_loc)
        
        # Link function or method to the file node or class node
        parent_node = self.current_class if self.current_class else self.current_file
        self.graph_builder.add_edge(parent_node, node_name, 'contains')
        
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        class_name = node.name
        docstring = ast.get_docstring(node)
        
        node_name = f"{self.current_file}::{class_name}"
        node_loc = (self.current_file, node.lineno, node.end_lineno)
        self.graph_builder.add_node(node_name, 'class', class_name, docstring, node_loc)
        
        # Link class to the file node
        self.graph_builder.add_edge(self.current_file, node_name, 'contains')
        
        previous_class = self.current_class
        self.current_class = node_name # use parent class as the current class
        
        self.generic_visit(node)
        # save the previous class
        self.current_class = previous_class

class ReferenceBuilder:
    def __init__(self, graph, function_definitions):
        self.graph = graph  # The graph produced by GraphBuilder
        self.function_definitions = function_definitions
    def build_references(self, repo_path):
        """Build reference edges between functions in the graph."""
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    rel_file_path = os.path.relpath(file_path, repo_path)
                    with open(file_path, "r") as file:
                        tree = ast.parse(file.read())
                        self._visit_tree(tree, rel_file_path)

    def _visit_tree(self, tree, file_path):
        """Visit the tree to find function definitions and references."""
        visitor = FunctionReferenceVisitor(self.graph, self.function_definitions, file_path)
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
        full_function_name = f"{self.current_file}::{function_name}" if self.current_class is None else f"{self.current_file}::{self.current_class}::{function_name}"
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

if __name__ == "__main__":
    # Example usage
    repo_path = "../../tests/test_repo"
    graph_builder = GraphBuilder(repo_path, save_jpg=True, out_file_name="repo_graph.jpg")
    graph_builder.build_whole_graph(repo_path)