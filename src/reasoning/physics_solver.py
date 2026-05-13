# src/reasoning/physics_solver.py
"""
Physics Solver — deterministic SymPy-based formula solver.

DESIGN CONTRACT:
  Primary: solve(world_model: WorldModel) -> ReasoningOutput
  Legacy:  solve_question(text: str) -> ReasoningOutput

Uses:
  1. VariableCanonicalizer for name bridging
  2. DerivationGraph for multi-step solving
  3. Single-formula fallback for simple problems
"""

import sympy as sp
from typing import Dict, Any, Optional

from src.reasoning.formula_bank import get_formula, rank_formulas
from src.reasoning.world_model import WorldModel, DerivedFact, Quantity
from src.reasoning.variable_canonicalizer import VariableCanonicalizer
from src.reasoning.derivation_graph import DerivationGraph
from src.data.schema import ReasoningOutput

_canonicalizer = VariableCanonicalizer()


class PhysicsSolver:

    def solve(self, wm: WorldModel) -> ReasoningOutput:
        """Primary entry point — accepts a WorldModel."""
        trace: list = []

        # 1. Flatten WorldModel → raw variable dict
        raw_vars = wm.flat_quantities()

        # Inject geometry-derived distances
        for rel in wm.relations:
            if rel.relation_type == "distance" and rel.value:
                key = "r" if rel.entities in (["generic"], []) else "_".join(rel.entities)
                raw_vars.setdefault(key, rel.value.value)
                raw_vars.setdefault("r", rel.value.value)

        # 2. Canonicalize variable names
        known_vars = _canonicalizer.normalize_for_formula(raw_vars)

        trace.append({
            "action": "canonicalize",
            "raw": list(raw_vars.keys()),
            "canonical": list(known_vars.keys()),
            "compilation_method": wm.compilation_method,
        })

        if not known_vars:
            return self._failure("No variables after canonicalization", trace, wm.raw_text)

        # 3. Determine target
        target = wm.primary_goal_target()
        if target:
            target = _canonicalizer.normalize_single(target)

        # 4. Try multi-body Coulomb if 3+ charges detected
        charges = {k: v for k, v in known_vars.items() if k.startswith("q") and k[1:].isdigit()}
        if len(charges) >= 2 and target in ("F", None):
            distances = {k: v for k, v in known_vars.items() if k.startswith("r")}
            if distances:
                dag = DerivationGraph()
                dag.set_known(known_vars)
                step = dag.solve_coulomb_multi(charges, distances, target="F")
                if step:
                    trace.extend(dag.get_trace())
                    answer_str = f"{step.result:.4g} N"
                    wm.derived_facts.append(DerivedFact(
                        fact_type="quantity", name="F",
                        value=Quantity(step.result, "N", "derived"),
                        derived_by="coulomb_multi",
                    ))
                    return ReasoningOutput(
                        question=wm.raw_text, answer=answer_str,
                        explanation=f"Multi-body Coulomb: {len(charges)} charges. Resultant: {answer_str}.",
                        reasoning_trace=trace, used_premises=[], confidence=0.95, valid=True,
                    )

        # 5. DAG-based multi-step solving
        dag = DerivationGraph()
        dag.set_known(known_vars)
        if target:
            dag.set_target(target)
        step = dag.solve()
        if step:
            trace.extend(dag.get_trace())
            answer_str = f"{step.result:.4g} {step.unit}".strip()
            wm.derived_facts.append(DerivedFact(
                fact_type="quantity", name=step.target,
                value=Quantity(step.result, step.unit, "derived"),
                derived_by=step.formula_id,
            ))
            return ReasoningOutput(
                question=wm.raw_text, answer=answer_str,
                explanation=f"DAG: {len(dag.steps)} step(s). {step.formula_name} → {answer_str}.",
                reasoning_trace=trace, used_premises=[], confidence=1.0, valid=True,
            )

        # 6. Single-formula fallback (rank_formulas)
        ranked = rank_formulas(wm.raw_text, known_vars)
        if not ranked:
            return self._failure(
                f"No formula matched. Known: {list(known_vars.keys())}",
                trace, wm.raw_text
            )

        for formula_id, score in ranked:
            formula = get_formula(formula_id)
            for c_name, c_val in formula.get("constants", {}).items():
                known_vars.setdefault(c_name, c_val)

            local_trace = list(trace)
            local_trace.append({"action": "formula_selected", "formula": formula["name"], "score": score})

            eff_target = target
            if not eff_target or eff_target not in formula.get("variables", []):
                missing = [v for v in formula.get("variables", []) if v not in known_vars]
                if len(missing) == 1:
                    eff_target = missing[0]
                else:
                    continue

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
                fact_type="quantity", name=eff_target,
                value=Quantity(numeric_val, unit, "derived"),
                derived_by=formula_id,
            ))
            return ReasoningOutput(
                question=wm.raw_text, answer=answer_str,
                explanation=f"Using {formula['name']}, solved {eff_target}. Result: {answer_str}.",
                reasoning_trace=local_trace, used_premises=[], confidence=1.0, valid=True,
            )

        return self._failure(
            f"No formula solved. Ranked: {[f for f, _ in ranked[:3]]}. Known: {list(known_vars.keys())}",
            trace, wm.raw_text
        )

    def solve_question(self, text: str) -> ReasoningOutput:
        """Legacy entry: accepts raw text."""
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
        subst = {local_dict[k]: v for k, v in known.items() if k != target and k in local_dict}
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
