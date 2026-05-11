# src/reasoning/llm_compiler.py
"""
Optional LLM Compiler — OFF by default.

The LLM is ONLY allowed to emit constrained JSON.
It is NOT allowed to produce final answers.

Output schema:
{
  "entities": [{"name": "q1", "type": "charge", "value": 6e-8, "unit": "C"}],
  "relations": [{"type": "distance", "entities": ["A","B"], "value": 0.08, "unit": "m"}],
  "goals": [{"type": "compute", "target": "F", "unit": "N"}],
  "premises_fol": ["∀x (WT(x) → O(x))"],
  "task_type": "physics|logic",
  "fol_target": "qualifies_for_scholarship(Sophia)"
}

The deterministic engine remains authoritative over the final answer.
"""

import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Constrained JSON prompt template
_PHYSICS_PROMPT = """You are a semantic compiler. Extract structured information from the physics problem below.
Output ONLY valid JSON matching this schema — no explanation, no prose:

{{
  "entities": [{{"name": str, "type": str, "value": float_or_null, "unit": str_or_null}}],
  "relations": [{{"type": str, "entities": [str], "value": float_or_null, "unit": str_or_null}}],
  "goals": [{{"type": "compute", "target": str, "unit": str_or_null}}],
  "task_type": "physics"
}}

Problem:
{problem}

JSON:"""

_LOGIC_PROMPT = """You are a semantic compiler. Convert the question below to a FOL target predicate.
Output ONLY valid JSON — no explanation:

{{
  "task_type": "entailment" | "mcq",
  "fol_target": "predicate(Entity)" or null if MCQ,
  "entities": [{{"name": str, "role": str}}]
}}

Question: {question}
Premises summary: {premises_summary}

JSON:"""


class LLMCompiler:
    """
    Wraps an LLM and issues constrained JSON prompts.
    Returns parsed dicts or None on failure.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.enabled = llm is not None

    def compile_physics(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            prompt = _PHYSICS_PROMPT.format(problem=text)
            raw = self.llm.generate(prompt, max_new_tokens=512, temperature=0.0)
            # Extract JSON block from output
            json_str = self._extract_json(raw)
            if json_str:
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"LLMCompiler.compile_physics failed: {e}")
        return None

    def compile_logic_target(self, question: str,
                              premises_fol: list) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            summary = "; ".join(premises_fol[:5])
            prompt = _LOGIC_PROMPT.format(
                question=question, premises_summary=summary)
            raw = self.llm.generate(prompt, max_new_tokens=256, temperature=0.0)
            json_str = self._extract_json(raw)
            if json_str:
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"LLMCompiler.compile_logic_target failed: {e}")
        return None

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract first {...} block from LLM output."""
        import re
        m = re.search(r'\{.*\}', text, re.DOTALL)
        return m.group(0) if m else None
