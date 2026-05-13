# src/reasoning/derivation_graph.py
"""
DerivationGraph — multi-step DAG-based physics planner.

Replaces single-formula solving with:
  1. Build a dependency graph of formulas and quantities
  2. Iteratively solve each formula that has exactly 1 unknown
  3. Feed derived values back into the graph
  4. Support multi-hop: Coulomb → F1, Coulomb → F2, Resultant → F_net

Capabilities:
  - 2-body and 3-body Coulomb
  - Resultant force (vector addition, law of cosines)
  - Capacitor energy
  - Ohm's law chains
  - Pythagorean geometry
"""

import math
import sympy as sp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from src.reasoning.formula_bank import FORMULA_BANK, get_formula
from src.reasoning.variable_canonicalizer import VariableCanonicalizer

_canon = VariableCanonicalizer()


@dataclass
class DerivedQuantity:
    """One quantity in the derivation graph."""
    name: str
    value: Optional[float] = None
    unit: str = ""
    source: str = ""        # "given" | formula_id | "geometry"
    depth: int = 0          # derivation depth


@dataclass
class DerivationStep:
    """One step in the derivation trace."""
    formula_id: str
    formula_name: str
    target: str
    known_inputs: Dict[str, float]
    result: float
    unit: str
    depth: int


class DerivationGraph:
    """
    DAG-based multi-step physics planner.

    Usage:
        dag = DerivationGraph()
        dag.set_known({"q1": 6e-8, "q2": -6e-8, "r": 0.08})
        dag.set_target("F")
        result = dag.solve()
    """

    def __init__(self, max_depth: int = 10):
        self.quantities: Dict[str, DerivedQuantity] = {}
        self.target: Optional[str] = None
        self.steps: List[DerivationStep] = []
        self.max_depth = max_depth

    def set_known(self, var_dict: Dict[str, float], source: str = "given"):
        """Load known quantities from WorldModel."""
        for name, value in var_dict.items():
            self.quantities[name] = DerivedQuantity(
                name=name, value=value, source=source, depth=0
            )

    def set_target(self, target: str):
        self.target = target

    def known_values(self) -> Dict[str, float]:
        return {n: q.value for n, q in self.quantities.items() if q.value is not None}

    def solve(self) -> Optional[DerivationStep]:
        """
        Iteratively solve formulas until target is found or no progress.
        Returns the final DerivationStep that produces the target, or None.
        """
        for depth in range(1, self.max_depth + 1):
            progress = False

            for fid, formula in FORMULA_BANK.items():
                known = self.known_values()

                # Inject constants
                for c_name, c_val in formula.get("constants", {}).items():
                    known.setdefault(c_name, c_val)

                fvars = formula.get("variables", [])
                missing = [v for v in fvars if v not in known]

                if len(missing) != 1:
                    continue

                target_var = missing[0]

                # Skip if already derived
                if target_var in self.quantities and self.quantities[target_var].value is not None:
                    continue

                # Solve
                result = self._sympy_solve(formula, known, target_var)
                if result is None:
                    continue

                unit = formula.get("units", {}).get(target_var, "")
                step = DerivationStep(
                    formula_id=fid,
                    formula_name=formula["name"],
                    target=target_var,
                    known_inputs={k: v for k, v in known.items() if k in fvars},
                    result=result,
                    unit=unit,
                    depth=depth,
                )
                self.steps.append(step)
                self.quantities[target_var] = DerivedQuantity(
                    name=target_var, value=result, unit=unit,
                    source=fid, depth=depth,
                )
                progress = True

                # Early exit if target found
                if target_var == self.target:
                    return step

            if not progress:
                break

        # Check if target was derived
        if self.target and self.target in self.quantities:
            q = self.quantities[self.target]
            if q.value is not None:
                for s in reversed(self.steps):
                    if s.target == self.target:
                        return s
        return None

    def solve_coulomb_multi(self, charges: Dict[str, float],
                            distances: Dict[str, float],
                            target: str = "F") -> Optional[DerivationStep]:
        """
        Multi-body Coulomb: compute individual pair forces then vector-add.

        charges: {"q1": val, "q2": val, "q3": val, ...}
        distances: {"r12": val, "r13": val, "r23": val, ...}
                   OR {"r": val} for uniform distance

        Returns a DerivationStep for the resultant force.
        """
        k = 8.9875517923e9
        charge_names = sorted(charges.keys())
        n = len(charge_names)

        if n < 2:
            return None

        pair_forces: List[float] = []

        # Compute all pair forces
        for i in range(n):
            for j in range(i + 1, n):
                qi = charges[charge_names[i]]
                qj = charges[charge_names[j]]
                # Find distance
                r_key = f"r{i+1}{j+1}"
                r_val = distances.get(r_key, distances.get("r"))
                if r_val is None or r_val == 0:
                    continue

                f_pair = k * abs(qi * qj) / (r_val ** 2)
                pair_name = f"F_{charge_names[i]}_{charge_names[j]}"
                self.quantities[pair_name] = DerivedQuantity(
                    name=pair_name, value=f_pair, unit="N",
                    source="coulomb_pair", depth=1,
                )
                pair_forces.append(f_pair)
                self.steps.append(DerivationStep(
                    formula_id="coulomb_law",
                    formula_name=f"Coulomb ({charge_names[i]},{charge_names[j]})",
                    target=pair_name,
                    known_inputs={charge_names[i]: qi, charge_names[j]: qj, "r": r_val, "k": k},
                    result=f_pair, unit="N", depth=1,
                ))

        if not pair_forces:
            return None

        # If exactly 2 forces → use Pythagorean or direct sum heuristic
        if len(pair_forces) == 1:
            self.quantities[target] = DerivedQuantity(
                name=target, value=pair_forces[0], unit="N",
                source="single_pair", depth=2,
            )
            return self.steps[-1]

        if len(pair_forces) == 2:
            f1, f2 = pair_forces
            # Check if we have an angle (theta)
            theta = self.known_values().get("theta")
            if theta is not None:
                f_res = math.sqrt(f1**2 + f2**2 + 2 * f1 * f2 * math.cos(theta))
            else:
                f_res = math.sqrt(f1**2 + f2**2)

            step = DerivationStep(
                formula_id="resultant_force",
                formula_name="Resultant Force (vector sum)",
                target=target,
                known_inputs={"F1": f1, "F2": f2, "theta": theta or 1.5708},
                result=f_res, unit="N", depth=2,
            )
            self.steps.append(step)
            self.quantities[target] = DerivedQuantity(
                name=target, value=f_res, unit="N",
                source="vector_sum", depth=2,
            )
            return step

        # 3+ forces: scalar magnitude sum approximation
        f_total = math.sqrt(sum(f**2 for f in pair_forces))
        step = DerivationStep(
            formula_id="vector_sum_n",
            formula_name=f"Vector Sum ({len(pair_forces)} forces)",
            target=target,
            known_inputs={f"F{i+1}": f for i, f in enumerate(pair_forces)},
            result=f_total, unit="N", depth=2,
        )
        self.steps.append(step)
        self.quantities[target] = DerivedQuantity(
            name=target, value=f_total, unit="N",
            source="vector_sum_n", depth=2,
        )
        return step

    def get_trace(self) -> List[Dict]:
        """Return human-readable derivation trace."""
        return [
            {
                "depth": s.depth,
                "formula": s.formula_name,
                "target": s.target,
                "inputs": s.known_inputs,
                "result": f"{s.result:.6g} {s.unit}",
            }
            for s in self.steps
        ]

    @staticmethod
    def _sympy_solve(formula: dict, known: dict, target: str) -> Optional[float]:
        fvars = formula.get("variables", [])
        local_dict = {v: sp.Symbol(v) for v in fvars}

        if target in formula.get("inverse_solve_forms", {}):
            expr = sp.sympify(formula["inverse_solve_forms"][target], locals=local_dict)
            subst = {local_dict[k]: v for k, v in known.items() if k in local_dict}
            try:
                return float(abs(expr.subs(subst).evalf()))
            except Exception:
                return None

        eq = sp.sympify(formula["equation"], locals=local_dict)
        subst = {local_dict[k]: v for k, v in known.items()
                 if k != target and k in local_dict}
        try:
            solutions = sp.solve(eq.subs(subst), local_dict[target])
            if not solutions:
                return None
            return float(abs(solutions[0].evalf()))
        except Exception:
            return None
