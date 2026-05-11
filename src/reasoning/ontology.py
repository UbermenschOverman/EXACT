# src/reasoning/ontology.py
"""
Central ontology for EXACT neuro-symbolic engine.

Provides:
 - Physical variable aliases (U → voltage, I → current, ...)
 - SI unit normalization
 - Logic predicate normalization (NL phrase → canonical predicate)
 - NL → FOL pattern matching for question targets
"""

import re
from typing import Optional, Dict, List, Tuple

# ──────────────────────────────────────────────
# PHYSICAL VARIABLE ALIASES
# Maps any surface form → canonical variable name used in FormulaBank
# ──────────────────────────────────────────────
PHYSICS_VARIABLE_ALIASES: Dict[str, str] = {
    # Voltage
    "U": "V", "u": "V", "voltage": "V", "potential": "V",
    "emf": "V", "v": "V",
    # Current
    "I": "current", "i": "current", "amperage": "current",
    # Charge
    "q": "charge", "Q": "charge",
    "q1": "q1", "q2": "q2", "q3": "q3",       # keep named charges
    "q_1": "q1", "q_2": "q2", "q_3": "q3",
    # Resistance
    "R": "R", "r": "R", "resistance": "R",
    # Power
    "P": "P", "p": "P", "power": "P", "wattage": "P",
    # Energy
    "E": "energy", "W": "energy", "energy": "energy", "work": "energy",
    # Capacitance
    "C": "capacitance", "c": "capacitance",
    # Force
    "F": "F", "f": "F", "force": "F",
    "F1": "F1", "F2": "F2", "F_1": "F1", "F_2": "F2",
    # Distance
    "r": "r", "d": "r", "distance": "r",
    # Angle
    "theta": "theta", "α": "theta", "φ": "theta",
}

# ──────────────────────────────────────────────
# SI UNIT PREFIX TABLE
# ──────────────────────────────────────────────
SI_PREFIXES: Dict[str, float] = {
    "T": 1e12, "tera": 1e12,
    "G": 1e9,  "giga": 1e9,
    "M": 1e6,  "mega": 1e6,
    "k": 1e3,  "kilo": 1e3,
    "c": 1e-2, "centi": 1e-2,
    "m": 1e-3, "milli": 1e-3,
    "μ": 1e-6, "u": 1e-6, "micro": 1e-6,
    "n": 1e-9, "nano": 1e-9,
    "p": 1e-12, "pico": 1e-12,
}

# ──────────────────────────────────────────────
# CANONICAL UNIT STRINGS (for display)
# ──────────────────────────────────────────────
CANONICAL_UNITS: Dict[str, str] = {
    "V": "V", "volt": "V", "volts": "V",
    "A": "A", "amp": "A", "ampere": "A", "amps": "A",
    "Ω": "Ω", "ohm": "Ω", "ohms": "Ω",
    "W": "W", "watt": "W", "watts": "W",
    "F": "F", "farad": "F", "farads": "F",
    "J": "J", "joule": "J", "joules": "J",
    "C": "C", "coulomb": "C", "coulombs": "C",
    "N": "N", "newton": "N", "newtons": "N",
    "m": "m", "meter": "m", "meters": "m",
}

# ──────────────────────────────────────────────
# LOGIC PREDICATE ALIASES
# NL surface form → canonical predicate symbol
# ──────────────────────────────────────────────
PREDICATE_ALIASES: Dict[str, str] = {
    # Academic
    "well-tested": "WT", "well tested": "WT",
    "optimized": "O", "optimised": "O",
    "follows pep8": "PEP8", "pep 8": "PEP8", "pep8": "PEP8",
    "easy to maintain": "EM",
    "well-structured": "WS", "well structured": "WS",
    "clean and readable": "CR", "clean readable": "CR",
    "best practices": "BP",
    "follows best practices": "BP",
    # Scholarship
    "qualifies for scholarship": "qualifies_for_scholarship",
    "scholarship": "qualifies_for_scholarship",
    "completed thesis": "completed_thesis",
    "completed the thesis": "completed_thesis",
    "maintained gpa": "maintained_GPA",
    "maintained the gpa": "maintained_GPA",
    "high gpa": "maintained_GPA",
    "attended seminars": "attended_seminars",
    "attended all seminars": "attended_seminars",
    # Biology/science predicates
    "is a mammal": "mammal",
    "warm-blooded": "warm_blooded", "warm blooded": "warm_blooded",
    "can fly": "can_fly", "capable of flight": "can_fly",
    "is a bird": "bird",
}

# ──────────────────────────────────────────────
# NL QUESTION → FOL TARGET PATTERNS
# Used by the deterministic translator in SemanticCompiler
# Pattern: (regex on lowercase question, template for FOL target)
# {ENTITY} is replaced by extracted proper noun
# ──────────────────────────────────────────────
NL_TO_FOL_PATTERNS: List[Tuple[str, str]] = [
    # "Does/Can X qualify for scholarship?"
    (r"does (\w+) qualify for (?:a |the )?scholarship", "qualifies_for_scholarship({ENTITY})"),
    (r"will (\w+) (?:receive|get) (?:a |the )?scholarship", "qualifies_for_scholarship({ENTITY})"),
    # "Is X optimized?"
    (r"is (\w+) optimized", "O({ENTITY})"),
    (r"is (\w+) well[- ]tested", "WT({ENTITY})"),
    (r"is (\w+) easy to maintain", "EM({ENTITY})"),
    # "Can X fly?"
    (r"can (\w+) fly", "can_fly({ENTITY})"),
    # "Is it true that all X are Y?" → ForAll
    (r"(?:is it true|does it follow) that all (\w+) (?:are|is) (\w+)", "∀x ({P1}(x))"),
    # "Which conclusion follows?" → MCQ
    (r"which (?:conclusion|statement) (?:follows|is correct|is true)", "__MCQ__"),
    (r"which of the following", "__MCQ__"),
    # "Does X follow that Y?" → entailment
    (r"does it follow that (.+)\?", "__ENTAILMENT__"),
]

def normalize_variable(name: str) -> str:
    """Return canonical variable name from alias."""
    return PHYSICS_VARIABLE_ALIASES.get(name, name)

def normalize_unit(unit: str) -> str:
    """Return canonical unit string."""
    return CANONICAL_UNITS.get(unit, unit)

def predicate_for_phrase(phrase: str) -> Optional[str]:
    """Return canonical predicate symbol for an NL phrase."""
    key = phrase.strip().lower()
    return PREDICATE_ALIASES.get(key)

def question_to_fol_target(question: str) -> Tuple[str, Optional[str]]:
    """
    Deterministically translate an NL question into:
     - task_type: "entailment" | "mcq" | "unknown"
     - fol_target: the canonical FOL string to prove, or None for MCQ
    """
    q_lower = question.lower().strip()

    for pattern, template in NL_TO_FOL_PATTERNS:
        m = re.search(pattern, q_lower)
        if m:
            if template == "__MCQ__":
                return "mcq", None
            if template == "__ENTAILMENT__":
                return "entailment", None  # will need LLM fallback

            # Extract entity from first capture group
            entity = m.group(1).capitalize() if m.lastindex and m.lastindex >= 1 else "x"
            fol = template.replace("{ENTITY}", entity)
            return "entailment", fol

    # Default: if question contains A./B./C. style options → MCQ
    if re.search(r"\b[A-D]\.", question):
        return "mcq", None

    return "unknown", None
