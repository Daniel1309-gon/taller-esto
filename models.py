import numpy as np
import pandas as pd

class BayesianNode:
    """
    Represents a node in a Bayesian Network.
    
    Attributes:
        name (str): Unique identifier for the node.
        parents (list): List of parent BayesianNode objects.
        children (list): List of child BayesianNode objects.
        cpt (pd.DataFrame): Conditional Probability Table.
    """
    def __init__(self, name):
        self.name = name
        self.parents = []
        self.children = []
        self.cpt = None

    def set_cpt(self, cpt_data):
        """
        Sets the CPT for the node.
        cpt_data should be a pandas DataFrame where columns are parent names and 'prob'.
        For a root node, parents will be empty.
        """
        self.cpt = cpt_data

    def __repr__(self):
        return f"Node({self.name})"

class BayesianNetwork:
    """
    Represents a Bayesian Network (Directed Acyclic Graph).
    """
    def __init__(self):
        self.nodes = {}  # Map name -> BayesianNode

    def add_node(self, node):
        if node.name not in self.nodes:
            self.nodes[node.name] = node

    def add_edge(self, parent_name, child_name):
        """Adds a directed edge from parent to child."""
        if parent_name not in self.nodes or child_name not in self.nodes:
            raise ValueError("Both nodes must exist in the network.")
        
        parent = self.nodes[parent_name]
        child = self.nodes[child_name]
        
        if child not in parent.children:
            parent.children.append(child)
        if parent not in child.parents:
            child.parents.append(parent)

    def get_topological_order(self):
        """
        Returns a list of node names in topological order using Kahn's algorithm.
        """
        in_degree = {name: len(node.parents) for name, node in self.nodes.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            u_name = queue.pop(0)
            order.append(u_name)
            
            for child in self.nodes[u_name].children:
                in_degree[child.name] -= 1
                if in_degree[child.name] == 0:
                    queue.append(child.name)

        if len(order) != len(self.nodes):
            raise ValueError("The graph contains a cycle; it is not a DAG.")
            
        return order

    def get_node(self, name):
        return self.nodes.get(name)
