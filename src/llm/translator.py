# src/llm/translator.py

from .prompt import build_fol_prompt
from src.utils.json_utils import safe_json_load


class Translator:
    def __init__(self, generator):
        self.generator = generator

    def to_fol(self, question: str):
        prompt = build_fol_prompt(question)
        output = self.generator.run(prompt)

        parsed = safe_json_load(output)

        if parsed and "fol" in parsed:
            return parsed["fol"]

        return output