import networkx as nx

def get_skeleton(G):
    skeleton = G.to_undirected()
    return skeleton


def get_v_structures(G):
    v_structures = set()
    for node in G.nodes:
        parents = list(G.predecessors(node))
        if len(parents) == 2:
            parent1, parent2 = parents
            if not G.has_edge(parent1, parent2) and not G.has_edge(parent2, parent1):
                v_structures.add(tuple(sorted([parent1, node, parent2])))  # V-structure (X -> Z <- Y)
    return v_structures

def markov_equivalence(G1, G2):
    """
    Two graphs are Markov Equivalent iff they have the same
    skeleton and same immoralities
    """
    if (G1.nodes != G2.nodes):
        return False
    
    skeleton1 = get_skeleton(G1)
    skeleton2 = get_skeleton(G2)

    if not nx.is_isomorphic(skeleton1, skeleton2):
        return False  
    
    # immoralities 
    v_structures1 = get_v_structures(G1)
    v_structures2 = get_v_structures(G2)
    
    if v_structures1 == v_structures2:
        return True 
    else:
        return False
