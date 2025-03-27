import networkx as nx
from probability import *
import logging
import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)



class CausalGraph:
    def __init__(self, expression: CausalProbability, causal_structure=None):
        """
        Initialize the causal graph using a symbolic representation.
        expression is type CausalProbability
        """
        assert isinstance(expression, CausalProbability)
        self.expression = expression
        self.causal_structure = causal_structure
        self.graph = self._build_graph()


    def _build_graph(self):
        """
        Convert the structured probability expression into a causal DAG.
        """
        G = nx.DiGraph()
        
        probability_expr = self.expression
        outcome = probability_expr.args[0]
        conditions = probability_expr.args[1:] if len(probability_expr.args) > 1 else []
        
        G.add_node(outcome)
        for condition in conditions:
            if isinstance(condition, Do):  
                intervention_var = condition.args[0]  
                G.add_node(intervention_var)
                G.add_edge(intervention_var, outcome)
            else:
                G.add_node(condition)
                G.add_edge(condition, outcome)
        
        if self.causal_structure:
            for parent, children in self.causal_structure.items():
                for child in children:
                    G.add_edge(parent, child)
        return G
    

    def draw(self):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph, seed=42)  
        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color='lightblue')
        nx.draw_networkx_edges(self.graph, pos, arrowsize=20, width=2, edge_color='black')
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight='bold')
        plt.axis("off")
        plt.show()


expr = "P(Y | do(X), Z)"
expr = CausalProbability.parse(expr)
causal_graph = CausalGraph(expr)
print("Parsed Expression:", causal_graph.expression)
causal_graph.draw()
