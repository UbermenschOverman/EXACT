# src/llm/planner.py

from .prompt import build_planner_prompt
from src.utils.json_utils import safe_json_load


class Planner:
    def __init__(self, generator):
        self.generator = generator

    def plan(self, question: str):
        prompt = build_planner_prompt(question)
        output = self.generator.run(prompt)

        parsed = safe_json_load(output)

        if parsed and "steps" in parsed:
            return parsed["steps"]

        return [output]  # fallback