"""
This files defines mathematical functions such as probability, expectations...
add as we extend our library.

https://docs.sympy.org/latest/guides/custom-functions.html
"""

import sympy as sp
import re

class Probability(sp.Function):
    def __new__(cls, *args):
        if len(args) == 0:
            raise ValueError("Probability requires at least one argument.")
        return super().__new__(cls, *args)
    
    def __str__(self):
        if len(self.args) == 1:
            return f'P({self.args[0]})'
        return f'P({self.args[0]} | {', '.join(map(str, self.args[1:]))})'
    
    def __repr__(self):
        return self.__str__()
    

class Do(sp.Function):
    def __new__(cls, var):
        return super().__new__(cls, var)
    
    def __str__(self):
        return f'do({self.args[0]})'
    
    def __repr__(self):
        return self.__str__()
    

class CausalProbability(Probability):
    """
    Represents something similar to P(Y | do(X), do(W), Z).
    """
    def __new__(cls, outcome, *conditions):
        if not isinstance(outcome, sp.Symbol):
            raise ValueError("Outcome must be a symbolic variable.")
        return super().__new__(cls, outcome, *conditions)
    
    def __str__(self):
        if len(self.args) == 1:
            return f'P({self.args[0]})'
        return f'P({self.args[0]} | {', '.join(map(str, self.args[1:]))})'
    
    @classmethod
    def parse(cls, expr_str):
        """
        - 'P(Y)' 
        - 'P(Y | X)' 
        - 'P(Y | do(X), Z)' 
        - 'P(Y | do(X), do(W), Z)'
        """
        expr_str = expr_str.replace(' ', '')
        # pattern = r"P\(\s*([\w]+)\s*\|\s*((?:do\(\s*[\w]+\s*\)(?:,\s*)?)*)((?:,\s*[\w]+)*)?\s*\)"
        pattern = r"P\(\s*([\w]+)\s*(?:\|\s*((?:do\(\s*[\w]+\s*\)(?:,\s*)?)*)((?:[\w]+(?:,\s*[\w]+)*)?))?\s*\)"

        match = re.match(pattern, expr_str)
        
        if not match:
            raise ValueError(f"Invalid format: {expr_str}")
        
        effect, do_part, obs_part = match.groups()
        print(f"Effect: {effect}, Interventions: {do_part}, Observed: {obs_part}")
        symbols = {}
        
        def get_symbol(name):
            if name not in symbols:
                symbols[name] = sp.Symbol(name)
            return symbols[name]
        
        outcome = get_symbol(effect)

        do_vars = []
        if do_part:
            do_pattern = r"do\(\s*([\w]+)\s*\)"
            do_vars = re.findall(do_pattern, do_part)
        
        obs_vars = []
        if obs_part:
            obs_part = obs_part.strip(",")
            if obs_part:
                obs_vars = [z.strip() for z in obs_part.split(",")]

        conditions = []
        for do_str in do_vars:
            conditions.append(Do(get_symbol(do_str)))
        
        for obs_str in obs_vars:
            conditions.append((get_symbol(obs_str)))
        
        return cls(outcome, *conditions)



def test_parser():
    test_cases = [
        'P(Y)',
        'P(Y | X)',
        'P(Y | do(X))',
        'P(Y | do(X), Z)',
        'P(Y | do(X), do(W), Z)'
    ]
    
    for case in test_cases:
        try:
            prob = CausalProbability.parse(case)
            print(f"Parsed: {case}")
            print(f"Result: {prob}")
            print()
        except ValueError as e:
            print(f"Error parsing {case}: {e}")
            print()

# test_parser()

# X, Y, W, Z = sp.symbols("X Y W Z")

# cp1 = CausalProbability(Y, Do(X), Z)  

# cp2 = CausalProbability(Y, Do(X), Do(W), Z)  

# print("expr 1:", cp1)  
# print("expr 2:", cp2) 


