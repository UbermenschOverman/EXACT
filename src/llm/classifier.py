# src/llm/classifier.py

"""
Question type classifier — routes questions to appropriate solver.
Uses LLM for classification with keyword fallback.
"""

import re
from src.utils.json_utils import safe_json_load


class QuestionClassifier:
    """Classifies questions as logic, physics, definition, or unknown."""

    PHYSICS_KEYWORDS = [
        "voltage", "current", "resistance", "power", "ohm",
        "capacitor", "circuit", "series", "parallel", "watt",
        "electric field", "charge", "force", "energy", "joule",
        "calculate", "compute", "v=", "i=", "r=", "p=",
        "ampere", "coulomb", "farad",
    ]

    LOGIC_KEYWORDS = [
        "if.*then", "implies", "therefore", "premise",
        "conclude", "all.*are", "some.*are", "no.*are",
        "entail", "deduce", "infer", "valid", "syllogism",
        "logically follows", "contradiction",
    ]

    DEFINITION_KEYWORDS = [
        "what is", "define", "explain", "describe",
        "what does.*mean", "definition of",
    ]

    def classify(self, question: str, generator=None) -> str:
        """
        Classify question type.

        Args:
            question: The question text
            generator: Optional LLM generator for LLM-based classification

        Returns: 'physics' | 'logic' | 'definition' | 'unknown'
        """
        # Try keyword-based first (fast, no LLM needed)
        keyword_result = self._classify_keywords(question)
        if keyword_result != "unknown":
            return keyword_result

        # Fall back to LLM classification if generator available
        if generator:
            return self._classify_llm(question, generator)

        return "unknown"

    def _classify_keywords(self, question: str) -> str:
        """Fast keyword-based classification."""
        q_lower = question.lower()

        # Check physics
        physics_score = sum(
            1 for kw in self.PHYSICS_KEYWORDS
            if kw in q_lower
        )

        # Check logic (regex for patterns like "if...then")
        logic_score = sum(
            1 for kw in self.LOGIC_KEYWORDS
            if re.search(kw, q_lower)
        )

        # Check definition
        definition_score = sum(
            1 for kw in self.DEFINITION_KEYWORDS
            if re.search(kw, q_lower)
        )

        scores = {
            "physics": physics_score,
            "logic": logic_score,
            "definition": definition_score,
        }

        best = max(scores, key=scores.get)

        if scores[best] > 0:
            return best

        return "unknown"

    def _classify_llm(self, question: str, generator) -> str:
        """LLM-based classification as fallback."""
        from src.llm.prompt import build_classifier_prompt

        prompt = build_classifier_prompt(question)
        output = generator.run(prompt, temperature=0.1)

        parsed = safe_json_load(output)
        if parsed and "type" in parsed:
            q_type = parsed["type"].lower().strip()
            if q_type in ("physics", "logic", "definition"):
                return q_type

        return "unknown"
