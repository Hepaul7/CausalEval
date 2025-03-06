import random
import pandas as pd

TEMPLATES = [
    "What is the average treatment effect of {Γ} on {Δ}?",
    "What is the effect of changing the treatment {Δ} from 0 to 1 on the outcome {Γ} while holding {Λ} constant at some value {λ}?",
    "What is the effect of changing the treatment {Δ} from 0 to 1 on the outcome {Γ} while holding {Λ} constant at some value {λ}?"
]

MATH_EXPRESSIONS = [
    "E[Δ | do(Γ=1)] - E[Δ | do(Γ=0)]",
    "E[Δ|do(Γ=1,Λ=λ)] - E[Δ|do(Γ=0,Λ=λ)]",
    "E[Δ_{Λ(λ)}|do(Γ = 1)] - E[Y|do(Δ = 0)]"
]

VARIABLES = [
    ("smoking", "lung cancer", "age"),
    ("exercise", "weight loss", "diet"),
    ("education", "income", "IQ"),
    ("vaccination", "infection risk", "immunity level"),
    ("advertising", "sales", "consumer preference")
]

def generate_questions(num_questions: int, variables: list, templates: list, math_expressions: list):
    data = []
    assert len(templates) == len(math_expressions)
    num_templates = len(templates)
    for _ in range(num_questions):  
        T, Y, X = random.choice(variables)
        x_value = random.randint(0, 188)  
        template_idx = random.randint(0, num_templates - 1)
        question_template = templates[template_idx]
        math_expr = math_expressions[template_idx]
        # question_template = random.choice(templates)
        # math_expr = random.choice(math_expressions)

        math_expr = (
            math_expr.replace("Δ", T)
            .replace("Γ", Y)
            .replace("Λ", X)
            .replace("λ", str(x_value))
        )

        math_expr = math_expr.replace(f"{X}(", f"{X}(")  

        question = question_template.format(Δ=T, Γ=Y, Λ=X, λ=x_value)

        data.append((question, math_expr))

    df = pd.DataFrame(data, columns=["Natural Language Question", "Mathematical Representation"])
    df.to_csv("complex_causal_questions_dataset.csv", index=False)

def main():
    generate_questions(10, variables=VARIABLES, templates=TEMPLATES, math_expressions=MATH_EXPRESSIONS)

if __name__ == "__main__":
    main()
