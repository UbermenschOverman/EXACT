# src/reasoning/ontology.py
"""
Central ontology for EXACT neuro-symbolic engine.

Provides:
 - Physical variable aliases (U → V, I → current, ...)
 - SI unit normalization
 - Logic predicate normalization (NL phrase → canonical predicate)
 - NL → FOL pattern matching for question targets
"""

import re
from typing import Optional, Dict, List, Tuple

# ──────────────────────────────────────────────
# PHYSICAL VARIABLE ALIASES
# ──────────────────────────────────────────────
PHYSICS_VARIABLE_ALIASES: Dict[str, str] = {
    "U": "V", "u": "V", "voltage": "V", "potential": "V",
    "emf": "V", "v": "V",
    "I": "current", "i": "current", "amperage": "current",
    "q": "charge", "Q": "charge",
    "q1": "q1", "q2": "q2", "q3": "q3",
    "q_1": "q1", "q_2": "q2", "q_3": "q3",
    "R": "R", "r": "R", "resistance": "R",
    "P": "P", "p": "P", "power": "P", "wattage": "P",
    "E": "energy", "W": "energy", "energy": "energy", "work": "energy",
    "C": "capacitance", "c": "capacitance",
    "F": "F", "f": "F", "force": "F",
    "F1": "F1", "F2": "F2", "F_1": "F1", "F_2": "F2",
    "d": "r", "distance": "r",
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
# CANONICAL UNIT STRINGS
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
# Extensive coverage of paraphrase variants.
# Key: lowercase NL surface phrase → canonical predicate symbol
# ──────────────────────────────────────────────
PREDICATE_ALIASES: Dict[str, str] = {
    # ── Python/Software predicates ──
    "well-tested": "WT", "well tested": "WT", "is well-tested": "WT",
    "has been tested": "WT", "thoroughly tested": "WT",
    "optimized": "O", "optimised": "O", "is optimized": "O",
    "has been optimized": "O", "performs optimally": "O",
    "follows pep8": "PEP8", "pep 8": "PEP8", "pep8": "PEP8",
    "follows pep 8": "PEP8", "complies with pep8": "PEP8",
    "adheres to pep 8": "PEP8", "follows pep 8 standards": "PEP8",
    "does not follow pep8": "¬PEP8", "does not follow pep 8": "¬PEP8",
    "easy to maintain": "EM", "is easy to maintain": "EM",
    "maintainable": "EM", "easily maintained": "EM",
    "well-structured": "WS", "well structured": "WS",
    "is well-structured": "WS", "is well structured": "WS",
    "structured properly": "WS",
    "clean and readable": "CR", "clean readable": "CR",
    "has clean and readable code": "CR", "has clean code": "CR",
    "readable": "CR", "clean code": "CR",
    "best practices": "BP", "follows best practices": "BP",
    "adheres to best practices": "BP", "uses best practices": "BP",

    # ── Scholarship predicates ──
    "qualifies for scholarship": "qualifies_for_scholarship",
    "qualifies for the scholarship": "qualifies_for_scholarship",
    "qualifies for the university scholarship": "qualifies_for_scholarship",
    "eligible for scholarship": "qualifies_for_scholarship",
    "eligible for the scholarship": "qualifies_for_scholarship",
    "eligible for a scholarship": "qualifies_for_scholarship",
    "eligible for the university scholarship": "qualifies_for_scholarship",
    "receives scholarship": "qualifies_for_scholarship",
    "receives a scholarship": "qualifies_for_scholarship",
    "receives the scholarship": "qualifies_for_scholarship",
    "awarded scholarship": "qualifies_for_scholarship",
    "awarded a scholarship": "qualifies_for_scholarship",
    "awarded the scholarship": "qualifies_for_scholarship",
    "awarded funding": "qualifies_for_scholarship",
    "is awarded funding": "qualifies_for_scholarship",
    "gets financial aid": "qualifies_for_scholarship",
    "can receive financial aid": "qualifies_for_scholarship",
    "qualifies for financial support": "qualifies_for_scholarship",

    # ── Educational achievement predicates ──
    "completed thesis": "completed_thesis",
    "completed the thesis": "completed_thesis",
    "submitted the thesis": "completed_thesis",
    "finished thesis": "completed_thesis",
    "maintained gpa": "maintained_GPA",
    "maintained the gpa": "maintained_GPA",
    "high gpa": "maintained_GPA",
    "has high gpa": "maintained_GPA",
    "maintained a high gpa": "maintained_GPA",
    "attended seminars": "attended_seminars",
    "attended all seminars": "attended_seminars",
    "completed community service": "completed_community_service",
    "completed the required community service": "completed_community_service",
    "completed community service hours": "completed_community_service",
    "awarded honors diploma": "awarded_honors_diploma",
    "awarded an honors diploma": "awarded_honors_diploma",
    "received honors diploma": "awarded_honors_diploma",
    "qualified for advanced courses": "qualified_for_advanced_courses",
    "eligible for the international program": "eligible_for_international_program",
    "eligible for international program": "eligible_for_international_program",

    # ── Biology predicates ──
    "is a mammal": "mammal", "is mammal": "mammal",
    "warm-blooded": "warm_blooded", "warm blooded": "warm_blooded",
    "is warm-blooded": "warm_blooded",
    "can fly": "can_fly", "capable of flight": "can_fly",
    "able to fly": "can_fly", "flies": "can_fly",
    "is a bird": "bird", "is bird": "bird",

    # ── General academic predicates ──
    "passed the exam": "passed_exam", "passed exams": "passed_exam",
    "passed the test": "passed_exam",
    "graduated": "graduated", "has graduated": "graduated",

    # ── Financial support / eligibility synonyms ──
    "eligible for financial support": "qualifies_for_scholarship",
    "eligible for financial aid": "qualifies_for_scholarship",
    "receives financial aid": "qualifies_for_scholarship",
    "gets financial support": "qualifies_for_scholarship",
    "qualifies for funding": "qualifies_for_scholarship",
    "eligible for funding": "qualifies_for_scholarship",
    "receives funding": "qualifies_for_scholarship",
    "awarded a grant": "qualifies_for_scholarship",
    "receives a grant": "qualifies_for_scholarship",
    "benefits from financial support": "qualifies_for_scholarship",
    "is supported financially": "qualifies_for_scholarship",
    "qualifies for financial support": "qualifies_for_scholarship",
    "receives financial support": "qualifies_for_scholarship",

    # ── Software security/performance predicates ──
    "is secure": "secure", "is secured": "secure", "secure": "secure",
    "is vulnerable": "vulnerable", "has vulnerabilities": "vulnerable",
    "is performant": "performant", "performs well": "performant",
    "high performance": "performant", "performs optimally": "performant",
    "is reliable": "reliable", "is dependable": "reliable",
    "is scalable": "scalable",
    "is maintainable": "EM", "is easy to maintain": "EM",
    "is deployed": "deployed", "has been deployed": "deployed",
    "passes tests": "WT", "is tested": "WT",
    "fails tests": "¬WT", "is untested": "¬WT",
}

def normalize_variable(name: str) -> str:
    return PHYSICS_VARIABLE_ALIASES.get(name, name)

def normalize_unit(unit: str) -> str:
    return CANONICAL_UNITS.get(unit, unit)

def canonicalize_predicate_phrase(text: str) -> Optional[str]:
    """
    Normalize an NL phrase to a canonical predicate.
    Supports: exact match → prefix match → suffix match.
    """
    key = text.strip().lower().rstrip('.')

    # 1. Exact match
    if key in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[key]

    # 2. Substring match (longest wins)
    best = None
    best_len = 0
    for phrase, pred in PREDICATE_ALIASES.items():
        if phrase in key and len(phrase) > best_len:
            best = pred
            best_len = len(phrase)
    if best:
        return best

    return None

# Backward-compat alias
predicate_for_phrase = canonicalize_predicate_phrase

# ──────────────────────────────────────────────
# NL QUESTION → FOL TARGET PATTERNS
# (Used by question_translator.py)
# ──────────────────────────────────────────────
NL_TO_FOL_PATTERNS: List[Tuple[str, str]] = [
    # Scholarship
    (r"does (\w+) qualify for (?:a |the |the university )?scholarship", "qualifies_for_scholarship({ENTITY})"),
    (r"will (\w+) (?:receive|get) (?:a |the )?scholarship", "qualifies_for_scholarship({ENTITY})"),
    (r"(?:is|can) (\w+) (?:eligible|qualify) for (?:a |the )?scholarship", "qualifies_for_scholarship({ENTITY})"),
    (r"can (\w+) (?:receive|get) (?:financial aid|funding)", "qualifies_for_scholarship({ENTITY})"),
    # Optimized
    (r"is (\w+) (?:project )?optimized", "O({ENTITY})"),
    (r"is (\w+) (?:well[- ])?tested", "WT({ENTITY})"),
    (r"is (\w+) easy to maintain", "EM({ENTITY})"),
    # Entailment: "does it follow that ..."
    (r"does it follow that if all (\w+) (?:projects? )?are (\w[^,?]+)", "∀x ({P1}(x))"),
    # Can X fly?
    (r"can (\w+) fly", "can_fly({ENTITY})"),
    (r"is (\w+) capable of flight", "can_fly({ENTITY})"),
    # ── MCQ detection ──
    (r"which (?:conclusion|statement|option|claim|inference) (?:follows|is (?:correct|true|strongest|best|valid|most logical))", "__MCQ__"),
    (r"which of the following", "__MCQ__"),
    (r"what (?:is|can be) (?:the strongest |the )?conclusion", "__MCQ__"),
    (r"which is the (?:strongest|best|correct|most (?:likely|correct|logical|valid))", "__MCQ__"),
    (r"which (?:best )?(?:describes|represents|explains|summarizes|captures)", "__MCQ__"),
    (r"based on (?:the above|the) premises.*which", "__MCQ__"),
    (r"select (?:the )?(?:strongest|best|correct|most logical)", "__MCQ__"),
    (r"choose (?:the )?(?:most|best|correct)", "__MCQ__"),
    (r"identify (?:the )?(?:strongest|best|correct)", "__MCQ__"),
    (r"which (?:of the )?(?:options?|statements?|conclusions?) (?:below )?(?:is|are) (?:correct|valid|supported|true)", "__MCQ__"),
    (r"what can be (?:inferred|concluded|deduced)", "__MCQ__"),
    (r"the (?:strongest|best|correct) conclusion", "__MCQ__"),
    # ── Fewest premises MCQ ──
    (r"fewest premises", "__MCQ__"),
    # ── Yes/No entailment ──
    (r"according to the premises", "__ENTAILMENT__"),
    (r"does it follow (?:that|from)", "__ENTAILMENT__"),
    (r"can (?:we|it) (?:conclude|infer|deduce) (?:that|from)", "__ENTAILMENT__"),
    (r"is it (?:true|correct|valid) (?:that|to say)", "__ENTAILMENT__"),
]

def question_to_fol_target(question: str) -> Tuple[str, Optional[str]]:
    """
    Deterministically translate NL question → (task_type, fol_target).
    Returns:
        task_type: "mcq" | "entailment" | "unknown"
        fol_target: FOL string, or None for MCQ
    """
    q_lower = question.lower().strip()

    for pattern, template in NL_TO_FOL_PATTERNS:
        m = re.search(pattern, q_lower)
        if m:
            if template == "__MCQ__":
                return "mcq", None
            if template == "__ENTAILMENT__":
                return "entailment", None

            entity = m.group(1).capitalize() if m.lastindex >= 1 else "x"
            fol = template.replace("{ENTITY}", entity)
            return "entailment", fol

    # Structural MCQ detection: A. / B. / C. options
    if re.search(r'\b[A-D]\.', question):
        return "mcq", None

    return "unknown", None
