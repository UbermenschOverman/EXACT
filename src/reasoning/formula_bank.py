# src/reasoning/formula_bank.py

"""
Registry of physics formulas for deterministic computation.
Each formula defines its equation, variables, units, and detection keywords.
"""

FORMULA_BANK = {
    "ohm_law": {
        "name": "Ohm's Law",
        "equation": "V - I * R",          # SymPy: set to 0
        "solve_forms": {
            "V": "I * R",
            "I": "V / R",
            "R": "V / I",
        },
        "variables": {
            "V": {"desc": "voltage", "unit": "V"},
            "I": {"desc": "current", "unit": "A"},
            "R": {"desc": "resistance", "unit": "Ω"},
        },
        "keywords": ["ohm", "voltage", "current", "resistance", "v=ir", "i=v/r"],
    },
    "power_vi": {
        "name": "Electric Power (V×I)",
        "equation": "P - V * I",
        "solve_forms": {
            "P": "V * I",
            "V": "P / I",
            "I": "P / V",
        },
        "variables": {
            "P": {"desc": "power", "unit": "W"},
            "V": {"desc": "voltage", "unit": "V"},
            "I": {"desc": "current", "unit": "A"},
        },
        "keywords": ["power", "watt", "p=vi"],
    },
    "power_ir": {
        "name": "Electric Power (I²R)",
        "equation": "P - I**2 * R",
        "solve_forms": {
            "P": "I**2 * R",
            "I": "sqrt(P / R)",
            "R": "P / I**2",
        },
        "variables": {
            "P": {"desc": "power", "unit": "W"},
            "I": {"desc": "current", "unit": "A"},
            "R": {"desc": "resistance", "unit": "Ω"},
        },
        "keywords": ["power", "i squared r", "i²r", "p=i²r", "p=i^2r"],
    },
    "capacitor_energy": {
        "name": "Capacitor Energy",
        "equation": "E - Rational(1,2) * C * V**2",
        "solve_forms": {
            "E": "Rational(1,2) * C * V**2",
            "C": "2 * E / V**2",
            "V": "sqrt(2 * E / C)",
        },
        "variables": {
            "E": {"desc": "energy", "unit": "J"},
            "C": {"desc": "capacitance", "unit": "F"},
            "V": {"desc": "voltage", "unit": "V"},
        },
        "keywords": ["capacitor", "capacitance", "stored energy", "e=1/2cv"],
    },
    "electric_field": {
        "name": "Electric Field",
        "equation": "E_f - F / q",
        "solve_forms": {
            "E_f": "F / q",
            "F": "E_f * q",
            "q": "F / E_f",
        },
        "variables": {
            "E_f": {"desc": "electric field", "unit": "N/C"},
            "F": {"desc": "force", "unit": "N"},
            "q": {"desc": "charge", "unit": "C"},
        },
        "keywords": ["electric field", "force", "charge", "e=f/q", "coulomb"],
    },
    "resistance_series": {
        "name": "Series Resistance",
        "equation": "R_total - (R1 + R2)",
        "solve_forms": {
            "R_total": "R1 + R2",
            "R1": "R_total - R2",
            "R2": "R_total - R1",
        },
        "variables": {
            "R_total": {"desc": "total resistance", "unit": "Ω"},
            "R1": {"desc": "resistance 1", "unit": "Ω"},
            "R2": {"desc": "resistance 2", "unit": "Ω"},
        },
        "keywords": ["series", "total resistance", "r1+r2", "r1 + r2"],
    },
    "resistance_parallel": {
        "name": "Parallel Resistance",
        "equation": "1/R_total - (1/R1 + 1/R2)",
        "solve_forms": {
            "R_total": "(R1 * R2) / (R1 + R2)",
        },
        "variables": {
            "R_total": {"desc": "total resistance", "unit": "Ω"},
            "R1": {"desc": "resistance 1", "unit": "Ω"},
            "R2": {"desc": "resistance 2", "unit": "Ω"},
        },
        "keywords": ["parallel", "1/r", "parallel resistance"],
    },
}


def get_formula(formula_id: str) -> dict:
    """Get a formula by ID."""
    return FORMULA_BANK.get(formula_id)


def detect_formula(question: str) -> str:
    """
    Detect which formula to use based on keywords in the question.
    Returns formula_id or empty string.
    """
    question_lower = question.lower()

    # Score each formula by keyword matches
    scores = {}
    for fid, formula in FORMULA_BANK.items():
        score = sum(1 for kw in formula["keywords"] if kw in question_lower)
        if score > 0:
            scores[fid] = score

    if not scores:
        return ""

    # Return the best match
    return max(scores, key=scores.get)


def list_formulas() -> list:
    """List all available formulas."""
    return [
        {"id": fid, "name": f["name"], "equation": f["equation"]}
        for fid, f in FORMULA_BANK.items()
    ]
