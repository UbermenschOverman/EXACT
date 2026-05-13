# src/reasoning/question_translator.py
"""
QuestionTranslator — deterministic NL → FOL target converter.

Strategy:
  1. Rule-based patterns (ontology-driven)
  2. Entity + predicate extraction heuristics
  3. Confidence scoring
  4. Optional LLM fallback (if provided)

Returns QuestionTranslationResult with explicit source and explanation.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.reasoning.ontology import (
    NL_TO_FOL_PATTERNS, PREDICATE_ALIASES, canonicalize_predicate_phrase
)


@dataclass
class QuestionTranslationResult:
    fol_target: Optional[str]         # e.g. "qualifies_for_scholarship(Sophia)"
    task_type: str                    # "entailment" | "mcq" | "unknown"
    confidence: float                 # 0.0 – 1.0
    source: str                       # "rule" | "heuristic" | "llm" | "failed"
    explanation: str                  # human-readable derivation log
    entities: List[str] = field(default_factory=list)


class QuestionTranslator:
    """
    Converts a natural language logic question into a FOL target string
    that LogicSolver can evaluate against the premise graph.
    """

    def __init__(self, llm_compiler=None):
        self.llm = llm_compiler

    def translate(self, question: str,
                  premises_fol: Optional[List[str]] = None) -> QuestionTranslationResult:
        """
        Main entry point.
        Returns a QuestionTranslationResult.
        """
        # --- Pass 1: Rule-based patterns ---
        result = self._rule_based(question)
        if result.confidence >= 0.9:
            return result

        # --- Pass 2: Entity + predicate heuristic ---
        heuristic = self._heuristic(question)
        if heuristic.confidence > result.confidence:
            result = heuristic

        # --- Pass 3: LLM fallback ---
        if result.confidence < 0.6 and self.llm:
            llm_result = self._llm_fallback(question, premises_fol or [])
            if llm_result and llm_result.confidence > result.confidence:
                result = llm_result

        return result

    # ── Pass 1: Rule-based ───────────────────────────────────────────

    def _rule_based(self, question: str) -> QuestionTranslationResult:
        q_lower = question.lower().strip()

        for pattern, template in NL_TO_FOL_PATTERNS:
            m = re.search(pattern, q_lower)
            if not m:
                continue

            if template == "__MCQ__":
                return QuestionTranslationResult(
                    fol_target=None,
                    task_type="mcq",
                    confidence=0.95,
                    source="rule",
                    explanation=f"MCQ detected via pattern: {pattern}",
                )
            if template == "__ENTAILMENT__":
                return QuestionTranslationResult(
                    fol_target=None,
                    task_type="entailment",
                    confidence=0.5,
                    source="rule",
                    explanation="Generic entailment question; FOL target not determinable.",
                )

            entity = m.group(1).capitalize() if m.lastindex >= 1 else "x"
            fol = template.replace("{ENTITY}", entity)
            return QuestionTranslationResult(
                fol_target=fol,
                task_type="entailment",
                confidence=0.92,
                source="rule",
                explanation=f"Matched pattern '{pattern}' → {fol}",
                entities=[entity],
            )

        # MCQ structural fallback
        if re.search(r'\b[A-D]\.', question):
            return QuestionTranslationResult(
                fol_target=None, task_type="mcq",
                confidence=0.85, source="rule",
                explanation="MCQ detected via A./B./C./D. option structure.",
            )

        return QuestionTranslationResult(
            fol_target=None, task_type="unknown",
            confidence=0.0, source="rule",
            explanation="No pattern matched.",
        )

    # ── Pass 2: Entity + Predicate Heuristic ─────────────────────────

    def _heuristic(self, question: str) -> QuestionTranslationResult:
        """
        Extract a proper-noun entity and match a predicate phrase.
        Handles: "Does Sophia qualify for the university scholarship?"
        """
        # Extract proper nouns (capitalized words not at sentence start)
        words = question.split()
        entities = []
        for i, w in enumerate(words):
            clean = re.sub(r'[^\w]', '', w)
            if clean and clean[0].isupper() and i > 0:
                entities.append(clean)

        if not entities:
            return QuestionTranslationResult(
                fol_target=None, task_type="unknown",
                confidence=0.0, source="heuristic",
                explanation="No proper noun entity found.",
            )

        entity = entities[0]
        q_lower = question.lower()

        # Try predicate matching against the sentence (minus entity)
        q_no_entity = q_lower.replace(entity.lower(), "").strip()

        # Try each predicate alias phrase as substring
        matched_pred = canonicalize_predicate_phrase(q_no_entity)

        if matched_pred and not matched_pred.startswith("¬"):
            fol = f"{matched_pred}({entity})"
            return QuestionTranslationResult(
                fol_target=fol,
                task_type="entailment",
                confidence=0.75,
                source="heuristic",
                explanation=f"Entity='{entity}', predicate='{matched_pred}' → {fol}",
                entities=[entity],
            )

        # Look for "does X follow" entailment
        if re.search(r'does it follow|is it true|according to', q_lower):
            return QuestionTranslationResult(
                fol_target=None, task_type="entailment",
                confidence=0.4, source="heuristic",
                explanation="Generic entailment question detected; FOL target extraction failed.",
            )

        return QuestionTranslationResult(
            fol_target=None, task_type="unknown",
            confidence=0.1, source="heuristic",
            explanation=f"Entity '{entity}' found but no predicate phrase matched.",
        )

    # ── Pass 3: LLM Fallback ─────────────────────────────────────────

    def _llm_fallback(self, question: str,
                       premises_fol: List[str]) -> Optional[QuestionTranslationResult]:
        if not self.llm:
            return None
        try:
            out = self.llm.compile_logic_target(question, premises_fol)
            if not out:
                return None
            fol = out.get("fol_target")
            task = out.get("task_type", "unknown")
            return QuestionTranslationResult(
                fol_target=fol,
                task_type=task,
                confidence=0.65,
                source="llm",
                explanation=f"LLM returned: {out}",
            )
        except Exception as e:
            return QuestionTranslationResult(
                fol_target=None, task_type="unknown",
                confidence=0.0, source="llm",
                explanation=f"LLM failed: {e}",
            )
