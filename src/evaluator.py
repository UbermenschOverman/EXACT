# src/evaluator.py

"""
Enhanced evaluator with 4 separate metrics:
P1 — Answer accuracy (with numerical tolerance)
P2 — Explanation quality
P3 — Reasoning depth
P4 — Trace completeness
"""

import re
import math
from typing import List, Dict, Optional


class Evaluator:
    def __init__(self, numerical_tolerance: float = 0.01):
        self.numerical_tolerance = numerical_tolerance

    def evaluate(self, predictions: List[Dict],
                 references: List[Dict]) -> Dict:
        """
        Evaluate predictions against references.

        Returns:
            {
                "P1_answer_accuracy": float,
                "P2_explanation_quality": float,
                "P3_reasoning_depth": float,
                "P4_trace_completeness": float,
                "total": int,
                "per_sample": List[dict],
            }
        """
        results = {
            "P1_answer_accuracy": 0,
            "P2_explanation_quality": 0,
            "P3_reasoning_depth": 0,
            "P4_trace_completeness": 0,
            "total": len(predictions),
            "per_sample": [],
        }

        for pred, ref in zip(predictions, references):
            sample_scores = {}

            # P1: answer accuracy
            p1 = self._match_answer(pred.get("answer"), ref.get("answer"))
            sample_scores["P1"] = p1
            results["P1_answer_accuracy"] += p1

            # P2: explanation quality
            p2 = self._score_explanation(pred.get("explanation", ""))
            sample_scores["P2"] = p2
            results["P2_explanation_quality"] += p2

            # P3: reasoning depth
            p3 = self._score_reasoning(pred)
            sample_scores["P3"] = p3
            results["P3_reasoning_depth"] += p3

            # P4: trace completeness
            p4 = self._score_trace_completeness(pred)
            sample_scores["P4"] = p4
            results["P4_trace_completeness"] += p4

            results["per_sample"].append({
                "question": pred.get("question", ""),
                "scores": sample_scores,
                "predicted_answer": pred.get("answer", ""),
                "reference_answer": ref.get("answer", ""),
            })

        # Normalize
        n = results["total"]
        if n > 0:
            results["P1_answer_accuracy"] /= n
            results["P2_explanation_quality"] /= n
            results["P3_reasoning_depth"] /= n
            results["P4_trace_completeness"] /= n

        return results

    def _match_answer(self, pred, ref) -> float:
        """
        Match with numerical tolerance.
        Returns 1.0 for match, 0.5 for partial, 0.0 for miss.
        """
        if not pred or not ref:
            return 0.0

        pred_str = str(pred).strip().lower()
        ref_str = str(ref).strip().lower()

        # Exact string match
        if pred_str == ref_str:
            return 1.0

        # Extract numeric values for comparison
        pred_num = self._extract_number(pred_str)
        ref_num = self._extract_number(ref_str)

        if pred_num is not None and ref_num is not None:
            if ref_num == 0:
                return 1.0 if pred_num == 0 else 0.0
            if abs(pred_num - ref_num) / max(abs(ref_num), 1e-10) < self.numerical_tolerance:
                return 1.0
            # Partial credit if close (within 10%)
            if abs(pred_num - ref_num) / max(abs(ref_num), 1e-10) < 0.1:
                return 0.5

        # Check if reference is contained in prediction
        if ref_str in pred_str:
            return 0.5

        return 0.0

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract first numeric value from string."""
        match = re.search(r'[-+]?[\d.]+(?:e[-+]?\d+)?', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None

    def _score_explanation(self, explanation: str) -> float:
        """
        Heuristic explanation quality score.
        Checks length, reasoning keywords, and structure.
        """
        if not explanation:
            return 0.0

        score = 0.0

        # Length check (graduated)
        length = len(explanation)
        if length > 200:
            score += 0.3
        elif length > 100:
            score += 0.2
        elif length > 50:
            score += 0.1

        # Reasoning keywords
        keywords = [
            "because", "therefore", "thus", "if", "then",
            "since", "hence", "given", "applying", "using",
            "formula", "law", "equation",
        ]
        keyword_count = sum(1 for k in keywords if k in explanation.lower())
        score += min(0.4, keyword_count * 0.1)

        # Structure indicators (numbered steps, bullet points)
        if re.search(r'\d+[.)]\s', explanation) or "step" in explanation.lower():
            score += 0.15

        # Has equation/formula
        if re.search(r'[A-Z]\s*=\s*[A-Z/\*]', explanation):
            score += 0.15

        return min(score, 1.0)

    def _score_reasoning(self, pred: Dict) -> float:
        """
        Reasoning depth score based on structured reasoning artifacts.
        """
        score = 0.0

        # Has FOL / formal representation
        if pred.get("fol") or pred.get("formula_used"):
            score += 0.3

        # Has proof steps
        proof = pred.get("proof", [])
        if proof:
            score += min(0.3, len(proof) * 0.05)

        # Has used premises
        premises = pred.get("used_premises", [])
        if premises:
            score += min(0.2, len(premises) * 0.05)

        # Verified
        if pred.get("valid"):
            score += 0.1

        # Has confidence
        confidence = pred.get("confidence", 0)
        if confidence > 0.5:
            score += 0.1

        return min(score, 1.0)

    def _score_trace_completeness(self, pred: Dict) -> float:
        """
        Check if reasoning trace has all required steps.
        Physics: identify → extract → solve → compute → conclude
        Logic: add_premise → add_rule → derive → conclude
        """
        trace = pred.get("reasoning_trace", [])
        if not trace:
            return 0.0

        actions = [t.get("action", "") for t in trace]
        q_type = pred.get("question_type", "unknown")

        if q_type == "physics":
            required = ["identify_formula", "extract_variables", "solve", "compute"]
            found = sum(1 for r in required if r in actions)
            return found / len(required)

        elif q_type == "logic":
            required = ["add_premise", "derive"]
            optional = ["add_rule", "check_target", "detect_contradiction"]
            found = sum(1 for r in required if r in actions)
            bonus = sum(0.1 for o in optional if o in actions)
            return min((found / max(len(required), 1)) + bonus, 1.0)

        else:
            # General — check for any structure
            if len(trace) >= 3:
                return 0.7
            elif len(trace) >= 1:
                return 0.3
            return 0.0