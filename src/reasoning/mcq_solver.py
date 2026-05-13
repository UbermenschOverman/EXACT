# src/reasoning/mcq_solver.py
"""
MCQ Solver — evaluates multiple-choice logic options deterministically.

Scoring strategy:
  1. Parse premise FOL + forward chain
  2. For each option: check derivability, contradiction, implication matching
  3. Score based on proof depth (fewer premises = stronger conclusion)
"""

import logging
from typing import Dict, List, Optional, Tuple

from src.reasoning.world_model import WorldModel, MCQOption
from src.reasoning.premise_graph import PremiseGraph
from src.reasoning.parser import FOLParser
from src.reasoning.fol import Negation, Predicate, Implication
from src.reasoning.ontology import PREDICATE_ALIASES, canonicalize_predicate_phrase

logger = logging.getLogger(__name__)


class MCQSolver:

    def __init__(self):
        self.fol_parser = FOLParser()

    def evaluate(self, wm: WorldModel) -> Tuple[Optional[MCQOption], Dict[str, float]]:
        """Evaluate MCQ options. Returns (best_option, {opt_id: score})."""
        mcq_goal = next((g for g in wm.goals if g.goal_type == "mcq"), None)
        if not mcq_goal or not mcq_goal.options:
            return None, {}

        # Build premise graph
        graph = PremiseGraph()
        for i, p_str in enumerate(wm.premises_fol):
            try:
                ast = self.fol_parser.parse_ast(p_str)
                graph.add_premise(f"P{i+1}", ast, is_given=True)
            except Exception as e:
                logger.debug(f"Premise parse error P{i+1}: {e}")
                continue
        try:
            graph.forward_chain()
        except Exception as e:
            logger.debug(f"Forward chain error: {e}")

        derivable_texts = {n.text for n in graph.premises.values()}

        scores: Dict[str, float] = {}
        best_option: Optional[MCQOption] = None
        best_score = -float("inf")

        for option in mcq_goal.options:
            score = self._score_option(option, derivable_texts, graph)
            scores[option.option_id] = score
            option.score = score
            if score > best_score:
                best_score = score
                best_option = option

        return best_option, scores

    def _score_option(self, option: MCQOption,
                      derivable_texts: set,
                      graph: PremiseGraph) -> float:
        score = 0.0

        # 1. If FOL string was canonicalized, check derivability
        if option.fol_str:
            # Direct derivability check
            for dt in derivable_texts:
                if option.fol_str in dt or dt in option.fol_str:
                    score += 10.0
                    break

            # Try to parse and match structurally
            try:
                ast = self.fol_parser.parse_ast(option.fol_str)
                ast_text = str(ast)
                if ast_text in derivable_texts:
                    score += 10.0
                neg_text = str(Negation(ast))
                if neg_text in derivable_texts:
                    score -= 10.0  # contradiction penalty
            except Exception:
                pass

        # 2. Predicate phrase matching against derivable set
        derivable_preds = set()
        for dt in derivable_texts:
            if "(" in dt:
                pred_name = dt.split("(")[0].lstrip("¬∀∃ ")
                derivable_preds.add(pred_name)

        opt_lower = option.text.lower()
        for phrase, pred in PREDICATE_ALIASES.items():
            if phrase in opt_lower and pred in derivable_preds:
                score += 5.0
                break

        # 3. Implication options: "If A then B" — check if A→B is in premises
        if option.fol_str and "→" in option.fol_str:
            for dt in derivable_texts:
                if "→" in dt:
                    # Rough structural match of implication direction
                    if any(tok in option.fol_str for tok in dt.split("→")[0].split("(") if len(tok) > 1):
                        score += 3.0
                        break

        # 4. If text mentions "not" → potential contraposition
        if "not " in opt_lower or "does not" in opt_lower:
            # Contrapositive reasoning bonus if we can match
            pred = canonicalize_predicate_phrase(opt_lower)
            if pred and pred.startswith("¬") and pred[1:] in derivable_preds:
                score += 4.0

        return score
