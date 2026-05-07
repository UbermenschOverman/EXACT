# src/llm/prompt.py

"""
Prompt templates for all LLM interactions.
Each prompt enforces strict JSON output format.
"""


def build_planner_prompt(question: str) -> str:
    return f"""You are a reasoning planner.

Break the problem into ordered logical steps.

Return STRICT JSON:
{{
  "steps": ["step1", "step2", ...]
}}

Question:
{question}"""


def build_fol_prompt(question: str) -> str:
    return f"""Convert to First-Order Logic.

Return STRICT JSON:
{{
  "fol": "..."
}}

Question:
{question}"""


def build_explainer_prompt(question: str, reasoning: str, answer: str) -> str:
    return f"""Explain the reasoning clearly and concisely.

Return STRICT JSON:
{{
  "explanation": "..."
}}

Question:
{question}

Reasoning:
{reasoning}

Answer:
{answer}"""


def build_classifier_prompt(question: str) -> str:
    return f"""Classify this question into one of these types:
- "physics": involves numerical computation, formulas, or physical quantities
- "logic": involves logical reasoning, premises, implications, or deduction
- "definition": asks for a definition, description, or explanation of a concept

Return STRICT JSON:
{{
  "type": "physics" or "logic" or "definition"
}}

Question:
{question}"""


def build_premise_extraction_prompt(question: str) -> str:
    return f"""Extract all premises and logical rules from this question.

For each premise, identify:
- The statement text
- Whether it is negated
- Any if-then implications

Return STRICT JSON:
{{
  "premises": [
    {{"id": "P1", "text": "...", "negation": false}},
    {{"id": "P2", "text": "...", "negation": false}}
  ],
  "rules": [
    {{"id": "R1", "conditions": ["P1"], "conclusion": "C1", "text": "If P1 then C1"}}
  ],
  "target": "What we need to determine"
}}

Question:
{question}"""


def build_variable_extraction_prompt(question: str) -> str:
    return f"""Extract physics variables, their values, and units from this question.
Also identify which variable needs to be solved for.

Return STRICT JSON:
{{
  "variables": {{
    "V": {{"value": 10, "unit": "V"}},
    "R": {{"value": 5, "unit": "Ω"}}
  }},
  "target": "I",
  "formula_hint": "ohm_law"
}}

Question:
{question}"""