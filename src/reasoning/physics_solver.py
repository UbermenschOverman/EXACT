# src/reasoning/physics_solver.py

"""
Deterministic physics solver using SymPy.
Replaces LLM arithmetic with symbolic computation.
"""

from typing import Dict, List, Any, Optional
from sympy import symbols, sympify, solve, sqrt, Rational, N
from sympy.core.sympify import SympifyError

from src.reasoning.formula_bank import FORMULA_BANK, detect_formula, get_formula
from src.reasoning.variable_extractor import VariableExtractor


class PhysicsSolver:
    """
    SymPy-based deterministic solver for physics problems.
    Uses formula bank + variable extraction for step-by-step computation.
    """

    def __init__(self):
        self.extractor = VariableExtractor()

    def solve_question(self, question: str,
                       formula_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Full solve pipeline:
        1. Detect formula
        2. Extract variables
        3. Determine target
        4. Solve symbolically
        5. Return computation trace

        Returns:
            {
                "answer": str,
                "numeric_value": float,
                "unit": str,
                "formula_used": str,
                "used_premises": List[str],
                "reasoning_trace": List[dict],
                "confidence": float,
            }
        """
        trace = []

        # 1. Extract variables
        known_vars = self.extractor.extract(question)

        trace.append({
            "step": 1,
            "action": "extract_variables",
            "detail": f"Extracted variables: {known_vars}",
            "known": known_vars,
        })

        if not known_vars:
            return self._failure("Could not extract any variables from question", trace)

        # 2. Detect formula
        if not formula_id:
            formula_id = detect_formula(question, known_vars)

        if not formula_id:
            return self._failure("Could not identify applicable formula", trace)

        formula = get_formula(formula_id)
        if not formula:
            return self._failure(f"Formula '{formula_id}' not found in bank", trace)

        trace.append({
            "step": 2,
            "action": "identify_formula",
            "detail": f"Identified: {formula['name']}",
            "formula_id": formula_id,
            "formula_name": formula["name"],
        })

        # 3. Determine target variable
        target = self.extractor.extract_target(question)

        if not target:
            # Infer: target is the variable in the formula not in known_vars
            formula_vars = set(formula["variables"].keys())
            unknown = formula_vars - set(known_vars.keys())
            if len(unknown) == 1:
                target = unknown.pop()
            elif len(unknown) == 0:
                return self._failure("All variables are known, nothing to solve", trace)
            else:
                # Try first unknown
                target = sorted(unknown)[0]

        trace.append({
            "step": 3,
            "action": "determine_target",
            "detail": f"Solving for: {target}",
            "target": target,
        })

        # 4. Solve using SymPy
        try:
            result = self._sympy_solve(formula, known_vars, target)
        except Exception as e:
            return self._failure(f"SymPy solve error: {str(e)}", trace)

        if result is None:
            return self._failure(f"Could not solve for {target}", trace)

        # Get unit for result
        target_info = formula["variables"].get(target, {})
        unit = target_info.get("unit", "")

        trace.append({
            "step": 4,
            "action": "solve",
            "detail": f"Solving equation: {target} = {formula['solve_forms'].get(target, '?')}",
            "equation": formula["solve_forms"].get(target, formula["equation"]),
            "target": target,
        })

        trace.append({
            "step": 5,
            "action": "compute",
            "detail": f"Substituting values: {known_vars} → {target} = {result}{unit}",
            "result": float(result),
            "unit": unit,
        })

        # Build used premises
        used_premises = [
            f"{formula['name']}: {formula['equation'].replace(' - ', ' = ').replace('- ', '= ')}",
        ]
        for var, val in known_vars.items():
            var_info = formula["variables"].get(var, {})
            var_unit = var_info.get("unit", "")
            used_premises.append(f"Given: {var} = {val}{var_unit}")

        # Format answer
        numeric_val = float(result)
        answer_str = f"{numeric_val:g}{unit}"

        confidence = 0.95  # deterministic computation = high confidence

        trace.append({
            "step": 6,
            "action": "conclude",
            "detail": f"Answer: {answer_str}",
            "answer": answer_str,
            "confidence": confidence,
        })

        return {
            "answer": answer_str,
            "numeric_value": numeric_val,
            "unit": unit,
            "formula_used": formula["name"],
            "formula_id": formula_id,
            "used_premises": used_premises,
            "reasoning_trace": trace,
            "confidence": confidence,
        }

    def _sympy_solve(self, formula: dict, known_vars: dict,
                     target: str) -> Optional[float]:
        """
        Use SymPy to solve for target variable.
        Uses pre-defined solve_forms for reliability.
        """
        solve_forms = formula.get("solve_forms", {})

        if target in solve_forms:
            # Use pre-defined solve form
            expr_str = solve_forms[target]

            # Create symbol namespace with known values
            namespace = {"sqrt": sqrt, "Rational": Rational}
            for var, val in known_vars.items():
                namespace[var] = val

            # Also add formula variables as symbols for any not yet known
            for var in formula["variables"]:
                if var not in namespace:
                    namespace[var] = symbols(var)

            try:
                result = sympify(expr_str, locals=namespace)
                result = N(result)  # numerical evaluation
                return float(result)
            except (SympifyError, TypeError, ValueError):
                pass

        # Fallback: solve the general equation
        try:
            all_vars = {v: symbols(v) for v in formula["variables"]}
            target_sym = all_vars[target]

            eq_str = formula["equation"]
            namespace = {"sqrt": sqrt, "Rational": Rational}
            namespace.update(all_vars)

            eq = sympify(eq_str, locals=namespace)

            # Substitute known values
            for var, val in known_vars.items():
                if var in all_vars:
                    eq = eq.subs(all_vars[var], val)

            solutions = solve(eq, target_sym)

            if solutions:
                result = N(solutions[0])
                return float(result)
        except Exception:
            pass

        return None

    def _failure(self, reason: str, trace: list) -> dict:
        """Return a failure result."""
        trace.append({
            "step": len(trace) + 1,
            "action": "error",
            "detail": reason,
        })
        return {
            "answer": "UNKNOWN",
            "numeric_value": None,
            "unit": "",
            "formula_used": "",
            "formula_id": "",
            "used_premises": [],
            "reasoning_trace": trace,
            "confidence": 0.0,
        }
