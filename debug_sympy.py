import sympy as sp
E = sp.Symbol('E')
eq = sp.sympify("E - 0.5 * C * (U**2)")
known = {'C': 100e-6, 'U': 30.0}
subst_dict = {sp.Symbol(k): v for k, v in known.items()}
eq_subbed = eq.subs(subst_dict)
solutions = sp.solve(eq_subbed, E)
print(solutions)
