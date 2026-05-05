# src/reasoning/solver.py

from sympy import symbols, Implies
from sympy.logic.boolalg import simplify_logic


class Solver:
    def solve(self, parsed_fol):
        """
        Simplified solver
        """

        if parsed_fol["type"] == "implication":
            A, B = symbols('A B')
            expr = Implies(A, B)
            return simplify_logic(expr)

        return parsed_fol["value"]