from lark import Lark
from unidecode import unidecode

class CausalGrammar:
    def __init__(self):
        grammar = """
            start: expr

            expr: expectation
            | probability
            | do_expr
            | summation
            | variable_with_subscript  // Modified variable rule
            | variable
            | binary_operation
            | "(" expr ")"

            expectation: "E" "[" expr ("|" conditionals)? "]" -> expectation
            probability: "P" "(" expr ("|" conditionals)? ")" -> probability

            do_expr: "do" "(" assignment ("," assignment)* ")" -> do_expr

            summation: "Σ" "_{" variable "}" expr -> summation 
            
            binary_operation: expr ("+"|"-"|"*"|"=") expr -> binary_op

            conditionals: (assignment | do_expr) ("," (assignment | do_expr))* 

            assignment: variable "=" value -> assignment

            variable_with_subscript: variable "{" variable ("(" NUMBER ")")? "}" -> variable_subscript

            variable: /[A-Za-z][A-Za-z_0-9]*/  

            value: NUMBER | variable | variable_with_subscript

            NUMBER: /\d+(\.\d+)?/

            %import common.WS
            %ignore WS
        """
        self.grammar = grammar

class LarkParser:
    def __init__(self, grammar: str, parser: str = "lalr"):
        """
        Initialize the parser with a specific grammar.
        
        Args:
            grammar (str): The Lark grammar string
            parser (str, optional): Parser type. Defaults to "lalr".
        """
        self.grammar = grammar
        self.parser = Lark(self.grammar, parser=parser)
    
    def parse(self, expression: str):
        """
        Parses the given expression and pretty prints the parse tree if
        syntax is valid, otherwise prints invalid syntax and returns None.
        
        Args:
            expression (str): The expression to parse
        
        Returns:
            Parse tree if valid, None otherwise
        """
        try:
            # expression = unidecode(expression)  
            tree = self.parser.parse(expression)
            print("Valid syntax:", tree.pretty())
            return tree
        except Exception as e:
            print("Invalid syntax:", e)
            return None

def main():
    grammar = CausalGrammar()
    parser = LarkParser(grammar=grammar.grammar)
    
    expressions = [
        "E[Y | do(T=1)] - E[Y | do(T=0)]",
        "E[Y|do(T = 1,X= x)] - E[Y|do(T = 0,X= x)]",
        "E[Y_{X(0)}|do(T = 1)] - E[Y|do(T = 0)]",
        "E[Y_{X(1)}|do(T = 0)] - E[Y|do(T = 0)]",
        "E[Y|T = 1]-E[Y|T = 0]",
        "E[Y|T = 1,X= x]-E[Y|T = 0,X= x]",
        "Σ_{x} P(X= x|T = 0)*(E[Y|T = 1,X= x] - E[Y|T = 0,X= x])",
        "Σ_{x} (P(X= x|T = 1) - P(X=x|T=0))*E[Y|T = 0,X= x]"
    ]
    
    for exp in expressions:
        print(f"\nParsing expression: {exp}")
        parser.parse(exp)

if __name__ == "__main__":
    main()