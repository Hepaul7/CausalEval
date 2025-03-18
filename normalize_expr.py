import re
import itertools
import networkx as nx
import sympy as sp
from itertools import chain
import logging


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def replace_variables(expr: str):
    """
    Given some causal expression, replace the variables
    with variables in greek letters.

    *assumes that P and E are not variables
    """
    fvs = [
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ",
    "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω"]
    
    variables = set(re.findall(r"\b(?!P\()(?!(E\[))([a-zA-Z_]+)\b", expr))
    
    mapping = {var: greek for var, greek in zip(variables, itertools.cycle(fvs))}
    for var, greek in mapping.items():
        expr = re.sub(rf"\b{var}\b", greek, expr)
    
    return expr, mapping


def to_standard_p(expr: str):
    """
    Converts the expression to standard probabilites using Do-Calculus Rules
    """
    pass


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

    logger.debug(f'func: expr_to_digraph interventions: {interventions}')
    logger.debug(f'func: expr_to_digraph confounders{confounders}')
    for x in interventions:
        x = x.replace('do(', '').replace(')', '')
        G.add_edge(x, effect)

    # stuck here... we do not know the unique structure of this?
    # Ill just have the nodes of like Z and W, and from a DAG, 
    # if the nodes have no edges to or from it ill add it to just like P(Y| Z, W) when
    # converting back to a causal expression

    # so this node will always be d-separated from other nodes?
    for z in confounders:
        G.add_node(z)

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
    terms = right_side.split(',')  # :: List[Str]
    logger.debug(terms)
    logger.debug(G.nodes)
    
    do_terms = []
    condition_terms = []
    
    for term in terms:
        if 'do(' in term:
            do_terms.append(term.replace('do(', ''))
        else:
            condition_terms.append(term)
    
    G_modified = G.copy()
    for do_var in do_terms:
        # remove all incoming edges into a do() term
        for predecessor in list(G.predecessors(do_var)):
            G_modified.remove_edge(predecessor, do_var)
    
    # check each conditoning variable to see if it can be removed
    removable_conditions = []
    for z in condition_terms:
        # check if outcome is independent of z 
        other_conditions = [c for c in condition_terms if c != z]
        

        if nx.is_d_separator(G_modified, outcome, z, set(do_terms) | set(other_conditions)):
            removable_conditions.append(z)
    
    if removable_conditions:
        new_conditions = [c for c in condition_terms if c not in removable_conditions]
        
        if do_terms and new_conditions:
            new_expression = f"P({outcome}|{','.join(['do(' + x + ')' for x in do_terms])},{','.join(new_conditions)})"
        elif do_terms:
            new_expression = f"P({outcome}|{','.join(['do(' + x + ')' for x in do_terms])})"
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
            do_terms.append(term.replace('do(', ''))
        else:
            condition_terms.append(term)
    
    modified_terms = []
    modified = False
    
    for do_var in do_terms:
        G_modified = G.copy()
        
        for predecessor in list(G.predecessors(do_var)):
            G_modified.remove_edge(predecessor, do_var)
        
        if nx.is_d_separator(G_modified, outcome, do_var, set(do_terms) - {do_var} | set(condition_terms)):
            condition_terms.append(do_var)
            modified = True
        else:
            modified_terms.append(f"do({do_var})")
    
    if modified:
        if modified_terms and condition_terms:
            new_expression = f"P({outcome}|{','.join(modified_terms)},{','.join(condition_terms)})"
        elif modified_terms:
            new_expression = f"P({outcome}|{','.join(modified_terms)})"
        elif condition_terms:
            new_expression = f"P({outcome}|{','.join(condition_terms)})"
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
            do_terms.append(term.replace('do(', ''))
        else:
            condition_terms.append(term)
    
    removable_interventions = []
    for z_var in do_terms:
        if len(do_terms) > 1 and z_var == do_terms[0]:
            continue
            
        G_modified = G.copy()
        
        # 1. Remove all arrows going out of X (all main interventions)
        for x_var in do_terms:
            if x_var != z_var:  # Don't modify Z, only X
                for successor in list(G.successors(x_var)):
                    G_modified.remove_edge(x_var, successor)
        
        # 2. Remove all arrows going into Z
        for predecessor in list(G.predecessors(z_var)):
            G_modified.remove_edge(predecessor, z_var)
            
        other_do_vars = [x for x in do_terms if x != z_var]
        
        if nx.is_d_separator(G_modified, outcome, z_var, set(other_do_vars) | set(condition_terms)):
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
    
    return expr


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
print(apply_rule_1(g, expr))