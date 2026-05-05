# src/llm/prompt.py

def build_planner_prompt(question: str) -> str:
    return f"""
You are a reasoning planner.

Break the problem into ordered logical steps.

Return STRICT JSON:
{{
  "steps": ["step1", "step2", ...]
}}

Question:
{question}
"""


def build_fol_prompt(question: str) -> str:
    return f"""
Convert to First-Order Logic.

Return STRICT JSON:
{{
  "fol": "..."
}}

Question:
{question}
"""


def build_explainer_prompt(question: str, reasoning: str, answer: str) -> str:
    return f"""
Explain the reasoning clearly.

Return STRICT JSON:
{{
  "explanation": "..."
}}

Question:
{question}

Reasoning:
{reasoning}

Answer:
{answer}
"""