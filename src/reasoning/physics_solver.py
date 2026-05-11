# src/reasoning/physics_solver.py
"""
Physics Solver — deterministic SymPy-based formula solver.

DESIGN CONTRACT:
  Primary: solve(world_model: WorldModel) -> ReasoningOutput
  Legacy:  solve_question(text: str) -> ReasoningOutput
           (internally builds WorldModel via SemanticCompiler)

Solvers MUST NOT parse raw text directly.
"""

import sympy as sp
from typing import Dict, Any, Optional

from src.reasoning.formula_bank import get_formula, rank_formulas
from src.reasoning.world_model import WorldModel, DerivedFact, Quantity
from src.data.schema import ReasoningOutput


class PhysicsSolver:

    def solve(self, wm: WorldModel) -> ReasoningOutput:
        """Primary entry point — accepts a WorldModel."""
        trace: list = []

        # 1. Flatten WorldModel into {name: float} for formula bank
        known_vars = wm.flat_quantities()

        # Inject distances from geometry relations
        for rel in wm.relations:
            if rel.relation_type == "distance" and rel.value:
                key = "_".join(rel.entities) if rel.entities != ["generic"] else "r"
                if key not in known_vars:
                    known_vars[key] = rel.value.value
                if "r" not in known_vars and rel.entities == ["generic"]:
                    known_vars["r"] = rel.value.value

        trace.append({
            "action": "world_model_vars",
            "detail": f"Flattened variables: {known_vars}",
            "compilation_method": wm.compilation_method,
        })

        if not known_vars:
            return self._failure("WorldModel contains no quantities", trace, wm.raw_text)

        # 2. Determine target from WorldModel goals
        target = wm.primary_goal_target()

        # 3. Rank formulas
        ranked = rank_formulas(wm.raw_text, known_vars)
        if not ranked:
            return self._failure("No applicable formula found", trace, wm.raw_text)

        for formula_id, score in ranked:
            formula = get_formula(formula_id)

            # Inject physical constants
            for c_name, c_val in formula.get("constants", {}).items():
                known_vars.setdefault(c_name, c_val)

            local_trace = list(trace)
            local_trace.append({
                "action": "formula_selected",
                "formula": formula["name"],
                "score": score,
            })

            # Infer target from missing variables if not given
            eff_target = target
            if not eff_target or eff_target not in formula.get("variables", []):
                missing = [v for v in formula.get("variables", [])
                           if v not in known_vars]
                if len(missing) == 1:
                    eff_target = missing[0]
                else:
                    continue

            local_trace.append({"action": "target_identified", "target": eff_target})

            # 4. Symbolic solve
            try:
                numeric_val = self._sympy_solve(formula, known_vars, eff_target)
                if numeric_val is None:
                    continue
            except Exception:
                continue

            unit = formula.get("units", {}).get(eff_target, "")
            answer_str = f"{numeric_val:.4g} {unit}".strip()

            local_trace.append({"action": "solved", "result": answer_str})

            wm.derived_facts.append(DerivedFact(
                fact_type="quantity",
                name=eff_target,
                value=Quantity(numeric_val, unit, "derived"),
                derived_by=formula_id,
            ))

            return ReasoningOutput(
                question=wm.raw_text,
                answer=answer_str,
                explanation=(
                    f"Using {formula['name']}, solved for {eff_target}. "
                    f"Known: {known_vars}. Result: {answer_str}."
                ),
                reasoning_trace=local_trace,
                used_premises=[],
                confidence=1.0,
                valid=True,
            )

        return self._failure("No formula produced a valid solution", trace, wm.raw_text)

    def solve_question(self, text: str) -> ReasoningOutput:
        """Legacy entry: accepts raw text, builds WorldModel internally."""
        from src.reasoning.semantic_compiler import SemanticCompiler
        wm = SemanticCompiler().compile_physics(text)
        return self.solve(wm)

    @staticmethod
    def _sympy_solve(formula: dict, known: dict, target: str) -> Optional[float]:
        local_dict = {v: sp.Symbol(v) for v in formula["variables"]}

        if target in formula.get("inverse_solve_forms", {}):
            expr = sp.sympify(formula["inverse_solve_forms"][target], locals=local_dict)
            subst = {local_dict[k]: v for k, v in known.items() if k in local_dict}
            return float(abs(expr.subs(subst).evalf()))

        eq = sp.sympify(formula["equation"], locals=local_dict)
        subst = {local_dict[k]: v for k, v in known.items()
                 if k != target and k in local_dict}
        solutions = sp.solve(eq.subs(subst), local_dict[target])
        if not solutions:
            return None
        return float(abs(solutions[0].evalf()))

    @staticmethod
    def _failure(reason: str, trace: list, question: str) -> ReasoningOutput:
        trace.append({"action": "error", "detail": reason})
        return ReasoningOutput(
            question=question, answer="Unknown",
            explanation=reason, reasoning_trace=trace,
            used_premises=[], confidence=0.0, valid=False,
        )
