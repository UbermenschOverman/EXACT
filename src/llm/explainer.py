# src/llm/explainer.py

from .prompt import build_explainer_prompt
from src.utils.json_utils import safe_json_load


class Explainer:
    def __init__(self, generator):
        self.generator = generator

    def explain(self, question: str, reasoning: str, answer: str):
        prompt = build_explainer_prompt(question, reasoning, answer)
        output = self.generator.run(prompt, temperature=0.3)

        parsed = safe_json_load(output)

        if parsed and "explanation" in parsed:
            return parsed["explanation"]

        return output