import random
import pandas as pd
from transformers import AutoModelForCausalLM,  AutoTokenizer

from huggingface_hub import login
login('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')

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
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
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
        example = "What is the effect of changing the treatment smoking from 0 to 1 on the outcome lung cancer while holding age constant at some value 27?,E[smoking|do(lung cancer=1,age=27)] - E[smoking|do(lung cancer=0,age=27)]"
        prompt = f"Given the question: {question}\n Only provide the mathematical expression with no extra text. For example: {example}"
        inputs = tokenizer(prompt, return_tensors='pt')
        model_output = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(model_output[0], skip_special_tokens=True)
        print(response)

        data.append((question, math_expr, response))

    df = pd.DataFrame(data, columns=["Natural Language Question", "y_true", "y_pred"])
    df.to_csv("causal_questions_dataset.csv", index=False)

def main():
    generate_questions(10, variables=VARIABLES, templates=TEMPLATES, math_expressions=MATH_EXPRESSIONS)

if __name__ == "__main__":
    main()
