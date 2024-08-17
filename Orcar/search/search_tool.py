import networkx as nx

class SearchManager:
    
    def __init__(self):
        self.history = []

    def set_search_tool(self, search_tool):
        self.search_tool = search_tool

    def search(self, keyword):
        return self.search_tool.search(keyword)
    

def search_by_rag(prompt):
    """
    Setup embedding for prompt, use vector similarity to search for similar code snippets
    """
    pass


def search_by_keyword(file, keyword):
    """
    Given a keyword, search the file for the keyword, return the line number
    """
        
    with open(file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if keyword in line:
                return i
    return None

def search_function_def(kg, func_name):
        """
        Given a function name, search the KG for the function definition, return the loc (filepath, line number, end line number)
        """

        # kg is a networkx graph
        for node in kg.nodes:
            if node.name == func_name:
                return node.loc
        return None
 
