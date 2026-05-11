# src/reasoning/formula_bank.py

from typing import Dict, List, Any, Tuple

FORMULA_BANK = {
    "ohm_law": {
        "name": "Ohm's Law",
        "variables": ["V", "current", "R"],
        "aliases": ["ohm", "v=ir"],
        "target_variables": ["V", "current", "R"],
        "units": {"V": "V", "current": "A", "R": "Ω"},
        "keywords": ["ohm", "voltage", "current", "resistance"],
        "equation": "V - current * R",
        "inverse_solve_forms": {
            "V": "current * R",
            "current": "V / R",
            "R": "V / current"
        }
    },
    "power_vi": {
        "name": "Electric Power (V, I)",
        "variables": ["P", "V", "current"],
        "aliases": ["p=vi"],
        "target_variables": ["P", "V", "current"],
        "units": {"P": "W", "V": "V", "current": "A"},
        "keywords": ["power", "watt", "voltage", "current"],
        "equation": "P - V * current",
        "inverse_solve_forms": {
            "P": "V * current",
            "V": "P / current",
            "current": "P / V"
        }
    },
    "capacitor_energy": {
        "name": "Energy Stored in Capacitor",
        "variables": ["energy", "capacitance", "V"],
        "aliases": ["e=0.5cu^2"],
        "target_variables": ["energy", "capacitance", "V"],
        "units": {"energy": "J", "capacitance": "F", "V": "V"},
        "keywords": ["capacitor", "energy", "stored", "capacitance"],
        "equation": "energy - 0.5 * capacitance * (V**2)",
        "inverse_solve_forms": {}
    },
    "coulomb_law": {
        "name": "Coulomb's Law",
        "variables": ["F", "k", "q1", "q2", "r"],
        "aliases": ["electrostatic force"],
        "target_variables": ["F", "q1", "q2", "r"],
        "units": {"F": "N", "q1": "C", "q2": "C", "r": "m"},
        "keywords": ["coulomb", "charge", "force", "placed", "apart"],
        "equation": "F - (k * abs(q1 * q2)) / (r**2)",
        "constants": {"k": 8.9875517923e9},
        "inverse_solve_forms": {}
    },
    "resultant_force_cosines": {
        "name": "Law of Cosines Resultant Force",
        "variables": ["F", "F1", "F2", "theta"],
        "aliases": ["vector addition", "resultant"],
        "target_variables": ["F", "F1", "F2", "theta"],
        "units": {"F": "N", "F1": "N", "F2": "N", "theta": "rad"},
        "keywords": ["resultant", "force", "vector", "addition", "acting on", "angle"],
        "equation": "F - sqrt(F1**2 + F2**2 + 2*F1*F2*cos(theta))",
        "inverse_solve_forms": {}
    },
    "right_triangle_electrostatics": {
        "name": "Right Triangle Electrostatics",
        "variables": ["F", "F1", "F2"],
        "aliases": ["pythagoras"],
        "target_variables": ["F"],
        "units": {"F": "N", "F1": "N", "F2": "N"},
        "keywords": ["right triangle", "orthogonal", "perpendicular"],
        "equation": "F - sqrt(F1**2 + F2**2)",
        "inverse_solve_forms": {
            "F": "sqrt(F1**2 + F2**2)"
        }
    },
    "equilateral_triangle_electrostatics": {
        "name": "Equilateral Triangle Resultant Force",
        "variables": ["F", "F1", "F2"],
        "aliases": ["equilateral 60 deg"],
        "target_variables": ["F"],
        "units": {"F": "N", "F1": "N", "F2": "N"},
        "keywords": ["equilateral", "triangle", "resultant"],
        "equation": "F - sqrt(F1**2 + F2**2 + F1*F2)", # since cos(60) = 0.5, 2*F1*F2*0.5 = F1*F2
        "inverse_solve_forms": {
            "F": "sqrt(F1**2 + F2**2 + F1*F2)"
        }
    }
}

def get_formula(formula_id: str) -> dict:
    return FORMULA_BANK.get(formula_id, {})

def score_formula(formula: dict, question: str, known_vars: dict) -> float:
    """
    Competition-grade formula scoring with overlap algorithms and confidence.
    """
    question_lower = question.lower()
    score = 0.0
    
    # Keyword overlap scoring
    for kw in formula.get("keywords", []):
        if kw in question_lower:
            score += 1.5
            
    # Alias exact match scoring
    for alias in formula.get("aliases", []):
        if alias in question_lower:
            score += 3.0
            
    # Variable matching (strong signal)
    matched_vars = sum(1 for v in known_vars if v in formula.get("variables", []))
    score += matched_vars * 2.0
    
    # Penalize if too many variables are missing
    total_vars = len(formula.get("variables", []))
    constants = len(formula.get("constants", {}))
    missing = total_vars - matched_vars - constants
    if missing > 1: # Usually we can only solve for 1 unknown
        score -= missing * 2.0
        
    return max(0.0, score)

def rank_formulas(question: str, known_vars: dict = None) -> List[Tuple[str, float]]:
    """
    Returns a sorted list of (formula_id, score) for multi-formula ranking.
    """
    known_vars = known_vars or {}
    
    ranked = []
    for fid, formula in FORMULA_BANK.items():
        score = score_formula(formula, question, known_vars)
        if score > 0:
            ranked.append((fid, score))
            
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
