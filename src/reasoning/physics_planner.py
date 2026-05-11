# src/reasoning/physics_planner.py

from typing import List, Dict, Tuple, Set
import sympy as sp
from src.reasoning.formula_bank import FORMULA_BANK
from src.reasoning.physics_ir import PhysicsProblem

class PhysicsPlanner:
    """
    Multi-Step Physics DAG.
    Chains formulas to derive intermediate variables required to reach a target.
    """
    
    def plan_derivation(self, problem: PhysicsProblem) -> List[Tuple[str, str]]:
        """
        Returns a list of steps: [(formula_id, target_variable)]
        to solve the problem.
        """
        if not problem.targets:
            return []
            
        final_target = problem.targets[0].name
        known_vars = set(problem.entities.keys())
        
        # Inject constants as known
        for formula in FORMULA_BANK.values():
            if "constants" in formula:
                for c in formula["constants"].keys():
                    known_vars.add(c)
        
        if final_target in known_vars:
            return []
            
        # Very simple backwards BFS for DAG
        queue = [(final_target, [])]
        visited_targets = set([final_target])
        
        while queue:
            current_target, current_path = queue.pop(0)
            
            for fid, formula in FORMULA_BANK.items():
                if current_target in formula.get("variables", []):
                    missing = [v for v in formula.get("variables", []) if v not in known_vars and v != current_target]
                    
                    if len(missing) == 0:
                        # We can solve this formula immediately!
                        new_path = [(fid, current_target)] + current_path
                        # We assume this solves everything backward.
                        # For a full DAG, we would add the derived target to known_vars and loop forward.
                        return new_path
                    elif len(missing) == 1:
                        # We need exactly one more variable to solve this. Add it to the queue.
                        next_target = missing[0]
                        if next_target not in visited_targets:
                            visited_targets.add(next_target)
                            queue.append((next_target, [(fid, current_target)] + current_path))
                            
        return []

    def execute_plan(self, problem: PhysicsProblem, plan: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Executes the derived DAG plan sequentially.
        """
        current_vars = {k: v.value for k, v in problem.entities.items() if v.value is not None}
        
        for fid, target in plan:
            formula = FORMULA_BANK[fid]
            # Inject constants
            if "constants" in formula:
                for c_name, c_val in formula["constants"].items():
                    if c_name not in current_vars:
                        current_vars[c_name] = c_val
                        
            try:
                if target in formula.get("inverse_solve_forms", {}):
                    eq_str = formula["inverse_solve_forms"][target]
                    local_dict = {v: sp.Symbol(v) for v in formula["variables"]}
                    expr = sp.sympify(eq_str, locals=local_dict)
                    subst_dict = {local_dict[k]: v for k, v in current_vars.items() if k in local_dict}
                    val = float(abs(expr.subs(subst_dict).evalf()))
                else:
                    eq_str = formula["equation"]
                    local_dict = {v: sp.Symbol(v) for v in formula["variables"]}
                    eq = sp.sympify(eq_str, locals=local_dict)
                    subst_dict = {local_dict[k]: v for k, v in current_vars.items() if k != target and k in local_dict}
                    solutions = sp.solve(eq.subs(subst_dict), local_dict[target])
                    if not solutions:
                        return None
                    val = float(abs(solutions[0].evalf()))
                    
                current_vars[target] = val
            except Exception:
                return None
                
        return current_vars
