# src/reasoning/variable_canonicalizer.py
"""
VariableCanonicalizer — bidirectional alias resolution between
WorldModel entity names and FormulaBank variable names.

The physics dataset uses:
  C = 100 μF, U = 30 V  → needs: capacitance, V
  R = 5 Ω, I = 2 A      → needs: R, current
  q1 = 6e-8, q2 = ...   → already matches coulomb_law (q1, q2)

FormulaBank uses:
  ohm_law:         V, current, R
  capacitor_energy: energy, capacitance, V
  coulomb_law:     F, k, q1, q2, r
  power_vi:        P, V, current
"""

from typing import Dict, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL PHYSICS MAP: surface_name → formula_variable_name
# Surface names come from text extraction; formula names match FORMULA_BANK.
# ─────────────────────────────────────────────────────────────────────────────
SURFACE_TO_FORMULA: Dict[str, str] = {
    # Voltage
    "U": "V", "u": "V", "voltage": "V", "potential": "V",
    "EMF": "V", "emf": "V",
    # Current
    "I": "current", "i": "current", "amperage": "current",
    # Resistance
    "resistance": "R",
    # Power
    "P": "P", "power": "P",
    # Energy
    "E": "energy", "W": "energy", "work": "energy",
    # Capacitance: 'C' is in dataset as capacitance variable
    "C": "capacitance", "capacitance": "capacitance",
    # Charges — Coulomb formula already uses q1/q2
    "q": "q1",    # bare q → treat as q1
    "Q": "q1",
    "q1": "q1", "q2": "q2", "q3": "q3",
    "q_1": "q1", "q_2": "q2", "q_3": "q3",
    "qA": "q1", "qB": "q2", "qC": "q3",
    # Force
    "F": "F", "force": "F", "F1": "F1", "F2": "F2",
    "F_1": "F1", "F_2": "F2",
    # Distance / radius
    "d": "r", "distance": "r",
    # Keep r as r
    "r": "r",
    # Angle
    "alpha": "theta", "α": "theta", "phi": "theta", "φ": "theta",
    "angle": "theta",
}

# Reverse map: formula_variable → canonical display name
FORMULA_TO_DISPLAY: Dict[str, str] = {v: k for k, v in SURFACE_TO_FORMULA.items()
                                        if not any(c.isdigit() for c in k)}


class VariableCanonicalizer:
    """
    Normalizes variable dictionaries for formula matching.

    Usage:
        canon = VariableCanonicalizer()
        normalized = canon.normalize_for_formula({"U": 30.0, "C": 1e-4})
        # → {"V": 30.0, "capacitance": 1e-4}
    """

    def normalize_for_formula(self, raw: Dict[str, float]) -> Dict[str, float]:
        """
        Convert WorldModel/extractor variable names to FormulaBank names.
        Preserves values; skips keys that don't map.
        """
        result: Dict[str, float] = {}
        for key, val in raw.items():
            canon = SURFACE_TO_FORMULA.get(key, key)
            # Don't override if already set (first writer wins)
            if canon not in result:
                result[canon] = val
        return result

    def normalize_single(self, name: str) -> str:
        """Normalize one variable name."""
        return SURFACE_TO_FORMULA.get(name, name)

    def reverse_lookup(self, formula_name: str) -> str:
        """Get display name from formula variable name."""
        return FORMULA_TO_DISPLAY.get(formula_name, formula_name)

    def normalize_with_units(
        self, raw: Dict[str, Tuple[float, Optional[str]]]
    ) -> Dict[str, Tuple[float, Optional[str]]]:
        """
        Normalize variables that carry (value, unit) tuples.
        """
        result: Dict[str, Tuple[float, Optional[str]]] = {}
        for key, (val, unit) in raw.items():
            canon = SURFACE_TO_FORMULA.get(key, key)
            if canon not in result:
                result[canon] = (val, unit)
        return result
