# src/evaluator.py

from typing import List, Dict
import re


class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, predictions: List[Dict], references: List[Dict]):
        results = {
            "P1_accuracy": 0,
            "P2_explanation": 0,
            "P3_reasoning": 0,
            "total": len(predictions)
        }

        for pred, ref in zip(predictions, references):

            # P1: answer accuracy
            if self._match_answer(pred.get("answer"), ref.get("answer")):
                results["P1_accuracy"] += 1

            # P2: explanation quality
            results["P2_explanation"] += self._score_explanation(
                pred.get("explanation", "")
            )

            # P3: reasoning depth
            results["P3_reasoning"] += self._score_reasoning(
                pred
            )

        # normalize
        n = results["total"]
        if n > 0:
            results["P1_accuracy"] /= n
            results["P2_explanation"] /= n
            results["P3_reasoning"] /= n

        return results

    def _match_answer(self, pred, ref):
        if not pred or not ref:
            return False

        return str(pred).strip().lower() == str(ref).strip().lower()

    def _score_explanation(self, explanation: str):
        """
        heuristic scoring:
        - length
        - presence of reasoning keywords
        """

        if not explanation:
            return 0

        score = 0

        # length check
        if len(explanation) > 50:
            score += 0.5

        # reasoning keywords
        keywords = ["because", "therefore", "thus", "if", "then"]
        if any(k in explanation.lower() for k in keywords):
            score += 0.5

        return min(score, 1.0)

    def _score_reasoning(self, pred: Dict):
        """
        check if system used structured reasoning
        """

        score = 0

        # has FOL
        if pred.get("fol"):
            score += 0.4

        # has proof steps
        if pred.get("proof"):
            score += 0.4

        # verified
        if pred.get("valid"):
            score += 0.2

        return min(score, 1.0)