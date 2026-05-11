# src/reasoning/physics_solver.py

import sympy as sp
from typing import Dict, Any
from src.reasoning.formula_bank import get_formula, rank_formulas
from src.reasoning.variable_extractor import VariableExtractor
from src.reasoning.geometry_helpers import GeometryHelper
from src.data.schema import ReasoningOutput

class PhysicsSolver:
    def __init__(self):
        self.extractor = VariableExtractor()

    def solve_question(self, question: str) -> ReasoningOutput:
        trace = []

        # 1. Extract variables
        known_vars = self.extractor.extract(question)
        
        # Apply geometry helpers for angles
        known_vars = GeometryHelper.infer_angles(question, known_vars)

        trace.append({
            "step": 1,
            "action": "extract_variables",
            "detail": f"Extracted variables: {known_vars}",
            "known": known_vars,
        })

        if not known_vars:
            return self._failure("Could not extract any variables from question", trace, question)

        # 2. Rank formulas
        ranked_formulas = rank_formulas(question, known_vars)

        if not ranked_formulas:
            return self._failure("Could not identify any applicable formula", trace, question)

        for formula_id, score in ranked_formulas:
            formula = get_formula(formula_id)
            
            # Inject constants
            if "constants" in formula:
                for c_name, c_val in formula["constants"].items():
                    if c_name not in known_vars:
                        known_vars[c_name] = c_val

            local_trace = list(trace) # copy trace for this attempt
            local_trace.append({
                "step": 2,
                "action": "identify_formula",
                "detail": f"Identified: {formula['name']} (score: {score})",
                "formula_id": formula_id,
                "formula_name": formula["name"],
            })

            # 3. Determine target variable
            target = self.extractor.extract_target(question)
            
            # If target extraction failed, infer it from the missing formula variable
            if target == "Unknown" or target not in formula.get("variables", []):
                missing = [v for v in formula.get("variables", []) if v not in known_vars]
                if len(missing) == 1:
                    target = missing[0]
                else:
                    continue # Try next formula

            local_trace.append({
                "step": 3,
                "action": "determine_target",
                "detail": f"Solving for: {target}",
                "target": target,
            })

            # 4. Symbolic Solve
            try:
                # Try inverse solve forms first to bypass SymPy if hardcoded
                if target in formula.get("inverse_solve_forms", {}):
                    equation_str = formula["inverse_solve_forms"][target]
                    local_dict = {v: sp.Symbol(v) for v in formula["variables"]}
                    expr = sp.sympify(equation_str, locals=local_dict)
                    subst_dict = {local_dict[k]: v for k, v in known_vars.items() if k in local_dict}
                    numeric_val = float(abs(expr.subs(subst_dict).evalf()))
                else:
                    equation_str = formula["equation"]
                    local_dict = {v: sp.Symbol(v) for v in formula["variables"]}
                    eq = sp.sympify(equation_str, locals=local_dict)
                    
                    subst_dict = {local_dict[k]: v for k, v in known_vars.items() if k != target and k in local_dict}
                    eq_subbed = eq.subs(subst_dict)
                    
                    target_sym = local_dict[target]
                    solutions = sp.solve(eq_subbed, target_sym)
                    
                    if not solutions:
                        continue # Try next formula
                        
                    numeric_val = float(abs(solutions[0].evalf()))

            except Exception as e:
                continue # Try next formula

            unit = formula.get("units", {}).get(target, "")
            answer_str = f"{numeric_val:.4g} {unit}".strip()

            local_trace.append({
                "step": 4,
                "action": "solve",
                "detail": f"Calculated {target} = {answer_str}",
                "result": numeric_val,
            })

            explanation = f"Using {formula['name']}, we solved for {target}. Substituting the known values yields {answer_str}."

            return ReasoningOutput(
                question=question,
                answer=answer_str,
                explanation=explanation,
                reasoning_trace=local_trace,
                used_premises=[],
                confidence=1.0,
                valid=True
            )

        return self._failure("Could not solve with any ranked formulas", trace, question)

    def _failure(self, reason: str, trace: list, question: str = "") -> ReasoningOutput:
        trace.append({
            "step": len(trace) + 1,
            "action": "error",
            "detail": reason,
        })
        return ReasoningOutput(
            question=question,
            answer="Unknown",
            explanation=reason,
            reasoning_trace=trace,
            used_premises=[],
            confidence=0.0,
            valid=False
        )
