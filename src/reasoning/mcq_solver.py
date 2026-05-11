# src/reasoning/mcq_solver.py
"""
MCQ Solver — evaluates multiple-choice logic options deterministically.

Uses ontology canonicalization to convert NL options to FOL strings,
then uses PremiseGraph to score derivability and contradiction.
"""

import logging
from typing import Dict, List, Optional, Tuple

from src.reasoning.world_model import WorldModel, MCQOption
from src.reasoning.premise_graph import PremiseGraph
from src.reasoning.parser import FOLParser
from src.reasoning.fol import Negation
from src.reasoning.ontology import PREDICATE_ALIASES

logger = logging.getLogger(__name__)


class MCQSolver:

    def __init__(self):
        self.fol_parser = FOLParser()

    def evaluate(self, wm: WorldModel) -> Tuple[Optional[MCQOption], Dict[str, float]]:
        """
        Evaluate MCQ options in the WorldModel.
        Returns (best_option, {option_id: score}).
        """
        # Find MCQ goal
        mcq_goal = next((g for g in wm.goals if g.goal_type == "mcq"), None)
        if not mcq_goal or not mcq_goal.options:
            return None, {}

        # Build premise graph once
        graph = PremiseGraph()
        for i, p_str in enumerate(wm.premises_fol):
            ast = self.fol_parser.parse_ast(p_str)
            graph.add_premise(f"P{i+1}", ast, is_given=True)
        graph.forward_chain()

        # Collect all derivable text representations
        derivable_texts = {n.text for n in graph.premises.values()}

        scores: Dict[str, float] = {}
        best_option: Optional[MCQOption] = None
        best_score = -float("inf")

        for option in mcq_goal.options:
            score = self._score_option(option, derivable_texts)
            scores[option.option_id] = score
            option.score = score
            if score > best_score:
                best_score = score
                best_option = option

        return best_option, scores

    def _score_option(self, option: MCQOption,
                       derivable_texts: set) -> float:
        score = 0.0

        # 1. If FOL string was canonicalized, check derivability directly
        if option.fol_str:
            try:
                ast = self.fol_parser.parse_ast(option.fol_str)
                ast_text = str(ast)
                if ast_text in derivable_texts:
                    score += 10.0
                # Check negation
                neg_text = str(Negation(ast))
                if neg_text in derivable_texts:
                    score -= 10.0
            except Exception:
                pass

        # 2. Keyword matching against derivable predicate names
        # Extract predicate names from derivable set
        derivable_preds = set()
        for dt in derivable_texts:
            # e.g. "WT(x)" → "WT"
            if "(" in dt:
                derivable_preds.add(dt.split("(")[0].lstrip("¬∀∃ "))

        # Check if option text mentions known derivable predicates
        opt_lower = option.text.lower()
        for phrase, pred in PREDICATE_ALIASES.items():
            if phrase in opt_lower and pred in derivable_preds:
                score += 5.0
                break

        return score
