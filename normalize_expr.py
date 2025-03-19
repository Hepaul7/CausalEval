import re
import itertools
import networkx as nx
import sympy as sp
from itertools import chain
import logging
import matplotlib.pyplot as plt


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def expr_to_digraph(expr: str):
    """
    Converts a causal expression to a DAG.
    - Plan: 
        1. Basic Probability Expressions P(X), P(Y|X), P(Y|X, Z)
        2. Interventions P(Y|do(X))
        3. Adding Confounders P(Y|do(X), Z)
        4. E[Y | do(T=1)] - E[Y | do(T=0)]
    """
    # expr = replace_variables(expr)
    G = nx.DiGraph()
    expr = expr.replace(" ", "") 

    # pattern = r"P\(\s*([\w]+)\s*\|\s*do\(\s*([\w, ]+)\s*\)(?:,\s*([\w, ]+))?\s*\)"
    pattern = r"P\(\s*([\w]+)\s*\|\s*((?:do\(\s*[\w]+\s*\)(?:,\s*)?)+)(?:,\s*([\w, ]+))?\s*\)"
    match = re.match(pattern, expr)

    if not match:
        raise ValueError(f"Invalid causal expression: {expr}")

    effect, intervention, confounders = match.groups()

    interventions = [x.strip() for x in intervention.split(",")] if intervention else []
    confounders = [z.strip() for z in confounders.split(",")] if confounders else []
    # logger.info(f"effect: {effect}, intervention: {intervention}, confounders {confounders} ")

    logger.debug(f'func: expr_to_digraph interventions: {interventions}')
    logger.debug(f'func: expr_to_digraph confounders{confounders}')
    for x in interventions:
        x = x.replace('do(', '').replace(')', '')
        G.add_edge(x, effect)

    # might be problematic... 
    for z in confounders:
        G.add_edge(z, effect)

    return G


def apply_rule_1(G, expression):
    """
    https://plato.stanford.edu/entries/causal-models/do-calculus.html

    Rule 1 (Insertion/deletion of observations)
    P(Y | do(X), Z, W) = P(Y | do(X), W) if Z is independent of y, 
    given X and potentially other variables W. 

    In the DAG, we remove all arrows going into X.

    Args:
        G: The DAG representing causal relationships.
        expression: The causal expression
    """
    expression = expression.replace(" ", "") 
    parts = expression.split('|')
    outcome = parts[0].replace('P(', '').replace(')', '')
    right_side = parts[1].replace(')', '')
    terms = right_side.split(',')
    
    do_terms = []
    condition_terms = []
    
    for term in terms:
        if 'do(' in term:
            do_terms.append(term.replace('do(', '').replace(')', ''))
        else:
            condition_terms.append(term)
    
    if not do_terms:
        return expression
    
    convertible_interventions = []
    
    for z_var in do_terms:
        if len(do_terms) > 1 and z_var == do_terms[0]:
            continue
            
        G_modified = G.copy()
        for do_var in do_terms:
            for predecessor in list(G.predecessors(do_var)):
                G_modified.remove_edge(predecessor, do_var)
        
        other_interventions = [x for x in do_terms if x != z_var]
        
        # if Y and Z are d-seperated by X \cup W in G* where G* = G - {UX:U\in V(G)}
        if nx.is_d_separator(G_modified, {outcome}, {z_var}, set(other_interventions) | set(condition_terms)):
            convertible_interventions.append(z_var)
    
    if convertible_interventions:
        new_do_terms = [x for x in do_terms if x not in convertible_interventions]
        new_conditions = condition_terms + convertible_interventions
        
        if new_do_terms and new_conditions:
            new_expression = f"P({outcome}|{','.join(['do(' + x + ')' for x in new_do_terms])},{','.join(new_conditions)})"
        elif new_do_terms:
            new_expression = f"P({outcome}|{','.join(['do(' + x + ')' for x in new_do_terms])})"
        elif new_conditions:
            new_expression = f"P({outcome}|{','.join(new_conditions)})"
        else:
            new_expression = f"P({outcome})"
        
        return new_expression
    else:
        return expression


def apply_rule_2(G, expression):
    """
    Rule 2 (Action/Observation Exchange)
    P(Y|do(X), do(Z), W) = P(Y|do(X), Z, W) if Y and Z are independent,
    given X and potentially other variables W.

    In the DAG, we remove the arrow going into X and out of Z
    (Generalization of the Back-Door Criteria)
    """
    expression = expression.replace(" ", "") 
    
    parts = expression.split('|')
    outcome = parts[0].replace('P(', '').replace(')', '')
    
    right_side = parts[1].replace(')', '')
    terms = right_side.split(',')
    
    do_terms = []
    condition_terms = []
    
    for term in terms:
        if 'do(' in term:
            do_terms.append(term.replace('do(', '').replace(')', ''))
        else:
            condition_terms.append(term)
    
    if not do_terms:
        return expression
    
    # here we try to convert each do(Z) to Z
    convertible_interventions = []
    
    for z_var in do_terms:
        if len(do_terms) > 1 and z_var == do_terms[0]:
            continue
            
        # G-{UX:U\in V(G)}
        G_modified = G.copy()
        for do_var in do_terms:
            for predecessor in list(G.predecessors(do_var)):
                G_modified.remove_edge(predecessor, do_var)

        #  G-{UX:U\in V(G)} - {ZU:U\in V(G)}
        for successor in list(G.successors(z_var)):
            G_modified.remove_edge(z_var, successor)
        
        # Other interventions (W)
        other_interventions = [x for x in do_terms if x != z_var]
        
        # if Y and Z are d-seperated by X \cup W in G** Where G** = G-{UX:U\in V(G)} - {ZU:U\in V(G)}
        if nx.is_d_separator(G_modified, {outcome}, {z_var}, set(other_interventions) | set(condition_terms)):
            convertible_interventions.append(z_var)
    
    if convertible_interventions:
        # Convert applicable do(Z) terms to observation Z
        new_do_terms = [x for x in do_terms if x not in convertible_interventions]
        new_conditions = condition_terms + convertible_interventions
        
        if new_do_terms and new_conditions:
            new_expression = f"P({outcome}|{','.join(['do(' + x + ')' for x in new_do_terms])},{','.join(new_conditions)})"
        elif new_do_terms:
            new_expression = f"P({outcome}|{','.join(['do(' + x + ')' for x in new_do_terms])})"
        elif new_conditions:
            new_expression = f"P({outcome}|{','.join(new_conditions)})"
        else:
            new_expression = f"P({outcome})"
        
        return new_expression
    else:
        return expression


def apply_rule_3(G, expression):
    """
    Rule 3 (Insertion/Deletion of Actions)
    P(Y|do(X), do(Z), W) = P(Y|do(X), W) if Y and Z are indepedent, given
    X and potentially other variables Z. 

    In the DAG we remove all arrow going out of X and all nodes of Z that are not 
    ancestors of W.
    """
    expression = expression.replace(" ", "")
    
    parts = expression.split('|')
    outcome = parts[0].replace('P(', '').replace(')', '')
    
    right_side = parts[1].replace(')', '')
    terms = right_side.split(',')
    
    do_terms = []
    condition_terms = []
    
    for term in terms:
        if 'do(' in term:
            do_terms.append(term.replace('do(', '').replace(')', ''))
        else:
            condition_terms.append(term)
    
    if len(do_terms) < 2:
        return expression
    
    removable_interventions = []
    
    primary_intervention = do_terms[0]
    secondary_interventions = do_terms[1:]
    
    for z_var in secondary_interventions:
        G_star = G.copy()
        
        # Create G* = G-{UX:U\in V(G)}
        for do_var in do_terms:
            for predecessor in list(G.predecessors(do_var)):
                G_modified.remove_edge(predecessor, do_var)
        
        ancestors_of_W = set()
        for w in condition_terms:
            if w in G_star.nodes:
                ancestors = nx.ancestors(G_star, w)
                ancestors_of_W.update(ancestors)
                ancestors_of_W.add(w) 

        # G** = G* - {UZ : U \in V(G*) \land UW \not in E(G*)}
        G_modified = G_star.copy()
        for predecessor in list(G_star.predecessors(z_var)):
            if predecessor not in ancestors_of_W:
                G_modified.remove_edge(predecessor, z_var)
        
        conditioning_set = set([primary_intervention]) | set(condition_terms)
        
        if nx.is_d_separator(G_modified, {outcome}, {z_var}, conditioning_set):
            removable_interventions.append(z_var)
    
    if removable_interventions:
        new_do_terms = [x for x in do_terms if x not in removable_interventions]
        
        if new_do_terms and condition_terms:
            new_expression = f"P({outcome}|{','.join(['do(' + x + ')' for x in new_do_terms])},{','.join(condition_terms)})"
        elif new_do_terms:
            new_expression = f"P({outcome}|{','.join(['do(' + x + ')' for x in new_do_terms])})"
        elif condition_terms:
            new_expression = f"P({outcome}|{','.join(condition_terms)})"
        else:
            new_expression = f"P({outcome})"
        
        return new_expression
    else:
        return expression



def simplify_expression(G, expr):
    prev_expr = None
    expr = expr.replace(' ', '')
    while prev_expr != expr:  
        prev_expr = expr
        expr = apply_rule_1(G, expr)
        expr = apply_rule_2(G, expr)
        expr = apply_rule_3(G, expr)
        expr = expr.replace(' ', '')
    return expr



def dag_to_causal_expression(G, outcome):
    """
    Converts a causal DAG to a standardized causal expression.
    Args:
        G: The DAG representing causal relationships.
        outcome: The target outcome variable

    TODO: check if the rules can be infinetly applied
    TODO: Completeness of Do-Calculus (if no rules can be further applied)
    """
    all_nodes = list(G.nodes)
    
    disconnected_nodes = [node for node in all_nodes 
                         if G.in_degree(node) == 0 and G.out_degree(node) == 0]
    
    direct_causes = list(G.predecessors(outcome))
    
    confounders = []
    for node in all_nodes:
        for cause in direct_causes:
            if G.has_edge(node, cause) and G.has_edge(node, outcome):
                confounders.append(node)
                break  
    
    other_observed = []
    for node in all_nodes:
        if (node != outcome and 
            node not in direct_causes and 
            node not in confounders and
            node not in disconnected_nodes):
            other_observed.append(node)
    
    if direct_causes:
        if confounders or other_observed or disconnected_nodes:
            conditioning = []
            if confounders:
                conditioning.extend(confounders)
            if other_observed:
                conditioning.extend(other_observed)
            if disconnected_nodes:
                conditioning.extend(disconnected_nodes)
                
            expr = f"P({outcome} | do({','.join(direct_causes)}), {','.join(conditioning)})"
        else:
            expr = f"P({outcome} | do({','.join(direct_causes)}))"
    else:
        conditioning = confounders + other_observed + disconnected_nodes
        if conditioning:
            expr = f"P({outcome} | {','.join(conditioning)})"
        else:
            expr = f"P({outcome})"
    
    expr = simplify_expression(expr)
    
    return expr


def draw_dag(G):
    plt.figure(figsize=(10, 7))
    
    pos = nx.spring_layout(G, seed=42)  
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, arrowsize=20, width=2, edge_color='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return plt

# expr = "P(Y | do(X, Z))"
# g = expr_to_digraph(expr)
# print(g.adj)
# print(dag_to_causal_expression(g, 'Y'))

# expr = "P(Y | do(X), Z, W)"
# g = expr_to_digraph(expr)
# apply_rule_1(g, expr)
# print(g.adj)
# print(g.nodes)
# print(dag_to_causal_expression(g, 'Y'))

# print(apply_rule_3(g, expr))



expr = "P(Y | do(X), do(Z), W)"
g = expr_to_digraph(expr)
print(f'ORIGINAL EXPR: {expr}, NORMALIZED EXPR: {apply_rule_2(g, expr)}')