# src/reasoning/solver.py
"""
EndToEndOrchestrator — single public entry point for all reasoning.

Pipeline:
  RAW TEXT / DICT
    ↓ SemanticCompiler (builds WorldModel)
    ↓ PhysicsSolver.solve(wm)  OR  LogicSolver.solve(wm)  OR  MCQSolver
    ↓ ReasoningOutput
"""

import logging
from typing import Any, Dict, Optional

from src.data.schema import ReasoningOutput
from src.reasoning.semantic_compiler import SemanticCompiler
from src.reasoning.world_model import WorldModel
from src.reasoning.physics_solver import PhysicsSolver
from src.reasoning.logic_solver import LogicSolver
from src.reasoning.mcq_solver import MCQSolver

logger = logging.getLogger(__name__)


class EndToEndOrchestrator:
    """
    The ONE authoritative orchestrator.
    All external callers (run.py, benchmarks, tests) use this.
    """

    def __init__(self, llm=None, use_llm_compiler: bool = False):
        llm_compiler = None
        if use_llm_compiler and llm is not None:
            from src.reasoning.llm_compiler import LLMCompiler
            llm_compiler = LLMCompiler(llm)

        self.compiler = SemanticCompiler(llm_compiler=llm_compiler)
        self.physics_solver = PhysicsSolver()
        self.logic_solver = LogicSolver()
        self.mcq_solver = MCQSolver()

    # ── Public entry ───────────────────────────────────────────────────

    def run(self, sample_data: Any) -> ReasoningOutput:
        """
        Route any input type through the full pipeline.
        Accepts: str (physics), dict (logic dataset item), Sample object.
        """
        if isinstance(sample_data, dict):
            return self._run_logic(sample_data)
        elif isinstance(sample_data, str):
            return self._run_physics(sample_data)
        elif hasattr(sample_data, "question"):
            return self._run_physics(sample_data.question)
        else:
            return ReasoningOutput(
                question=str(sample_data), answer="Unknown",
                explanation="Unknown input type", valid=False,
            )

    def run_physics(self, text: str) -> ReasoningOutput:
        return self._run_physics(text)

    def run_logic(self, item: Dict[str, Any]) -> ReasoningOutput:
        return self._run_logic(item)

    # ── Internal routing ───────────────────────────────────────────────

    def _run_physics(self, text: str) -> ReasoningOutput:
        wm = self.compiler.compile_physics(text)
        return self.physics_solver.solve(wm)

    def _run_logic(self, item: Dict[str, Any]) -> ReasoningOutput:
        wm = self.compiler.compile_logic(item)
        questions = item.get("questions", [""])
        q_text = questions[0] if questions else ""

        if not wm.goals:
            return ReasoningOutput(
                question=q_text, answer="Unknown",
                explanation="SemanticCompiler could not identify task goal",
                valid=False, confidence=0.0,
            )

        goal = wm.goals[0]

        if goal.goal_type == "mcq":
            best, scores = self.mcq_solver.evaluate(wm)
            answer = best.option_id if best else "Unknown"
            explanation = (f"MCQ scored: {scores}. "
                          f"Best: {best.text if best else 'None'}")
            return ReasoningOutput(
                question=q_text,
                answer=answer,
                explanation=explanation,
                reasoning_trace=[{"action": "mcq_eval", "scores": scores}],
                valid=best is not None,
                confidence=0.8 if best else 0.0,
            )

        elif goal.goal_type == "entailment":
            return self.logic_solver.solve(wm.premises_fol, goal.target)

        return ReasoningOutput(
            question=q_text, answer="Unknown",
            explanation=f"Unsupported goal type: {goal.goal_type}",
            valid=False,
        )