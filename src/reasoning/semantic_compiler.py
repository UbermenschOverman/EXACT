# src/reasoning/semantic_compiler.py
"""
SemanticCompiler — the central grounding layer for EXACT.

Pipeline:
  RAW TEXT
    ↓ Layer A: Structural Extraction (improved regex + token windows)
    ↓ Layer B: Ontology Normalization
    ↓ Layer C: LLM Compiler (optional fallback, OFF by default)
  WorldModel

The SemanticCompiler is the ONLY component allowed to touch raw text.
After it returns a WorldModel, all downstream components operate
exclusively on that structured representation.
"""

import re
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from src.reasoning.world_model import WorldModel, Quantity, MCQOption, Goal
from src.reasoning.ontology import (
    SI_PREFIXES, PHYSICS_VARIABLE_ALIASES, CANONICAL_UNITS,
    PREDICATE_ALIASES, question_to_fol_target, normalize_variable
)
from src.reasoning.physics_scene_builder import PhysicsSceneBuilder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER A helpers — structural extraction
# ─────────────────────────────────────────────────────────────────────────────

class StructuralExtractor:
    """
    Layer A: token-window and regex extraction.
    Handles both:
      (a) explicit format: "q1 = 6 × 10^-8 C"
      (b) narrative format: "a charge of 6 × 10^-8 C at point A"
    """

    # Comprehensive value pattern:
    # Matches: 6e-8, 6×10^-8, 6*10^-8, 6x10^-8, 6·10⁻⁸, 100, 100.5
    _VALUE = (
        r'([-+]?\d+(?:\.\d+)?)'                       # base number
        r'(?:'
        r'\s*[×x\*·]\s*10\s*[\^]\s*([-+]?\d+)'       # × 10^exp
        r'|[eE]([-+]?\d+)'                             # e-notation
        r'|'                                            # superscript unicode
        r'\s*[×x\*·]\s*10\s*((?:[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))'
        r')?'
    )
    _SUPERS = {'⁻': '-', '⁺': '+', '⁰': '0', '¹': '1', '²': '2', '³': '3',
               '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}

    _UNIT_PATTERNS = [
        (r'C|coulomb(?:s)?',   'C'),
        (r'V|volt(?:s)?',      'V'),
        (r'A|amp(?:ere)?(?:s)?', 'A'),
        (r'Ω|ohm(?:s)?',       'Ω'),
        (r'W|watt(?:s)?',      'W'),
        (r'F|farad(?:s)?',     'F'),
        (r'J|joule(?:s)?',     'J'),
        (r'N|newton(?:s)?',    'N'),
        (r'm|meter(?:s)?',     'm'),
    ]

    def _parse_value(self, base: str, exp_hat: str,
                     exp_e: str, exp_uni: str) -> float:
        val = float(base)
        exp = 0
        if exp_hat:
            exp = int(exp_hat)
        elif exp_e:
            exp = int(exp_e)
        elif exp_uni:
            exp = int(''.join(self._SUPERS.get(c, c) for c in exp_uni))
        return val * (10 ** exp)

    def extract_quantities(self, text: str) -> List[Tuple[str, float, str]]:
        """
        Returns list of (raw_name, numeric_value, canonical_unit).
        Handles both 'q1 = 6e-8 C' and 'a charge of 6e-8 C'.
        """
        results: List[Tuple[str, float, str]] = []

        for unit_pat, canon_unit in self._UNIT_PATTERNS:
            prefix_pat = r'([TGMkmμuncp]?)'

            # (1) Explicit: VarName = Value [prefix]Unit
            explicit = (
                r'([A-Za-z][A-Za-z0-9_]*)\s*=\s*'
                + self._VALUE + r'\s*' + prefix_pat + r'(' + unit_pat + r')\b'
            )
            for m in re.finditer(explicit, text):
                raw_name = m.group(1)
                val = self._parse_value(m.group(2), m.group(3), m.group(4), m.group(5))
                prefix = m.group(6)
                if prefix and prefix in SI_PREFIXES:
                    val *= SI_PREFIXES[prefix]
                canon_name = normalize_variable(raw_name)
                results.append((canon_name, val, canon_unit))

            # (2) Narrative: "a charge of X [prefix]Unit" / "X [prefix]Unit charge"
            # Common NL patterns for each unit type
            if canon_unit == 'C':
                narrative_labels = ['charge', 'q']
            elif canon_unit == 'V':
                narrative_labels = ['voltage', 'potential', 'emf']
            elif canon_unit == 'Ω':
                narrative_labels = ['resistance', 'resistor']
            elif canon_unit == 'F':
                narrative_labels = ['capacitance', 'capacitor']
            elif canon_unit == 'N':
                narrative_labels = ['force']
            elif canon_unit == 'm':
                narrative_labels = []  # distances handled separately
            else:
                narrative_labels = []

            for label in narrative_labels:
                narr = (
                    r'(?:' + label + r'(?:\s+of)?|of\s+' + label + r')\s*'
                    + self._VALUE + r'\s*' + prefix_pat + r'(' + unit_pat + r')\b'
                )
                for m in re.finditer(narr, text, re.IGNORECASE):
                    val = self._parse_value(m.group(1), m.group(2), m.group(3), m.group(4))
                    prefix = m.group(5)
                    if prefix and prefix in SI_PREFIXES:
                        val *= SI_PREFIXES[prefix]
                    # Counter-based naming for multiple narrative quantities
                    results.append((label, val, canon_unit))

        return results

    def extract_named_charges(self, text: str) -> List[Tuple[str, float, str]]:
        """
        Specifically extract 'q1 = 6 × 10^-8 C', 'qA = 5 μC', etc.
        Named charges preserve their index (q1, q2, qA, qB).
        """
        results = []
        pattern = (
            r'\b(q[A-Za-z0-9]+)\s*=\s*'
            + self._VALUE
            + r'\s*([TGMkmμuncp]?)\s*C\b'
        )
        for m in re.finditer(pattern, text):
            name = m.group(1)
            val = self._parse_value(m.group(2), m.group(3), m.group(4), m.group(5))
            prefix = m.group(6)
            if prefix and prefix in SI_PREFIXES:
                val *= SI_PREFIXES[prefix]
            results.append((name, val, 'C'))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# LAYER B — Ontology normalization
# ─────────────────────────────────────────────────────────────────────────────

class OntologyNormalizer:
    """
    Layer B: map extracted surface forms to canonical ontology entries.
    """

    def detect_target(self, text: str) -> Optional[str]:
        """Heuristic target detection from question text."""
        t = text.lower()
        if any(k in t for k in ["force", "resultant", "net force"]):
            return "F"
        if any(k in t for k in ["energy", "stored energy", "work"]):
            return "energy"
        if any(k in t for k in ["capacitance"]):
            return "capacitance"
        if any(k in t for k in ["voltage", "potential difference", "emf"]):
            return "V"
        if any(k in t for k in ["current"]):
            return "current"
        if any(k in t for k in ["resistance"]):
            return "R"
        if any(k in t for k in ["charge", "find q", "determine q"]):
            return "charge"
        if any(k in t for k in ["power"]):
            return "P"
        return None

    def normalize_predicate_phrase(self, phrase: str) -> Optional[str]:
        """Return canonical FOL predicate for an NL phrase."""
        key = phrase.lower().strip()
        return PREDICATE_ALIASES.get(key)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SEMANTIC COMPILER
# ─────────────────────────────────────────────────────────────────────────────

class SemanticCompiler:
    """
    Central grounding layer.

    Usage:
        compiler = SemanticCompiler(llm_compiler=None)   # pure deterministic
        wm = compiler.compile_physics("Two charges q1=6e-8 C...")
        wm = compiler.compile_logic(dataset_item)
    """

    def __init__(self, llm_compiler=None):
        self.extractor = StructuralExtractor()
        self.normalizer = OntologyNormalizer()
        self.scene_builder = PhysicsSceneBuilder()
        self.llm = llm_compiler   # None → LLM path is skipped

    # ── Physics compilation ────────────────────────────────────────────

    def compile_physics(self, text: str) -> WorldModel:
        wm = WorldModel(raw_text=text, problem_type="physics")
        method = "structural"

        # Layer A — structural extraction
        named_charges = self.extractor.extract_named_charges(text)
        all_quantities = self.extractor.extract_quantities(text)

        # Populate named charges as separate entities
        charge_counter = 1
        for name, val, unit in named_charges:
            wm.set_quantity(name, "value", val, unit, "explicit")

        # Populate remaining quantities
        # De-duplicate: if a named charge q1 already covers a value, skip generic
        added_generic_charge = False
        for raw_name, val, unit in all_quantities:
            if raw_name in ("charge", "q") and named_charges:
                continue  # named charges already handled
            ent_name = raw_name if raw_name not in ("charge", "q") else f"q{charge_counter}"
            if raw_name in ("charge", "q"):
                charge_counter += 1
            wm.set_quantity(ent_name, "value", val, unit, "explicit")

        # Layer B — detect goal / target
        target = self.normalizer.detect_target(text)
        if target:
            wm.goals.append(Goal("compute", target))

        # Layer C — optional LLM enrichment
        if self.llm and not wm.entities:
            llm_out = self.llm.compile_physics(text)
            if llm_out:
                self._apply_llm_physics(wm, llm_out)
                method = "llm"
            else:
                method = "structural_fallback"
        else:
            method = "structural" if wm.entities else "failed"

        wm.compilation_method = method
        wm.compilation_confidence = 1.0 if wm.entities else 0.0

        # Scene builder — geometry enrichment
        self.scene_builder.enrich(wm)

        return wm

    def _apply_llm_physics(self, wm: WorldModel, llm_out: Dict[str, Any]):
        """Merge LLM JSON output into WorldModel."""
        for ent in llm_out.get("entities", []):
            name = ent.get("name", "unknown")
            val = ent.get("value")
            unit = ent.get("unit")
            if val is not None:
                wm.set_quantity(name, "value", float(val), unit, "llm")
        for rel in llm_out.get("relations", []):
            val = rel.get("value")
            q = Quantity(float(val), rel.get("unit")) if val is not None else None
            wm.add_relation(rel.get("type", "unknown"), rel.get("entities", []), q)
        for goal in llm_out.get("goals", []):
            wm.goals.append(Goal(goal.get("type", "compute"),
                                 goal.get("target", "unknown"),
                                 goal.get("unit")))

    # ── Logic compilation ──────────────────────────────────────────────

    def compile_logic(self, item: Dict[str, Any]) -> WorldModel:
        """
        Build a WorldModel from a logic dataset item.
        item has keys: premises-FOL, premises-NL, questions, answers
        """
        wm = WorldModel(problem_type="logic")

        # Premises are pre-FOL in this dataset — directly use them
        premises_fol: List[str] = item.get("premises-FOL", [])
        wm.premises_fol = premises_fol
        wm.raw_text = "; ".join(item.get("premises-NL", []))

        questions: List[str] = item.get("questions", [])
        if not questions:
            return wm

        # Process each question into a Goal
        for q_text in questions:
            task_type, fol_target = question_to_fol_target(q_text)
            wm.raw_text = q_text  # last question is the context

            if task_type == "mcq":
                goal = Goal("mcq", "__mcq__")
                # Extract MCQ options A/B/C/D
                for m in re.finditer(r'\b([A-D])\.\s*(.+?)(?=\s+[A-D]\.|$)',
                                     q_text, re.DOTALL):
                    opt_id, opt_text = m.group(1), m.group(2).strip()
                    # Try to canonicalize option text to FOL
                    fol_opt = self._canonicalize_option(opt_text)
                    goal.options.append(MCQOption(opt_id, opt_text, fol_opt))
                wm.goals.append(goal)

            elif task_type == "entailment" and fol_target:
                wm.goals.append(Goal("entailment", fol_target))

            else:
                # LLM fallback for unknown task types
                if self.llm:
                    llm_out = self.llm.compile_logic_target(q_text, premises_fol)
                    if llm_out:
                        t = llm_out.get("task_type", "unknown")
                        ft = llm_out.get("fol_target")
                        if ft:
                            wm.goals.append(Goal("entailment", ft))
                        elif t == "mcq":
                            wm.goals.append(Goal("mcq", "__mcq__"))

        wm.compilation_method = "structural+ontology" if wm.goals else "failed"
        wm.compilation_confidence = 1.0 if wm.goals else 0.0
        return wm

    def _canonicalize_option(self, option_text: str) -> Optional[str]:
        """
        Convert an NL MCQ option to a FOL string using the ontology.
        Example: "Sophia qualifies for a scholarship" → "qualifies_for_scholarship(Sophia)"
        """
        # Check for named entity + predicate phrase
        # Pattern: "<Proper Noun> <predicate phrase>"
        m = re.match(r'([A-Z][a-z]+)\s+(.+)', option_text)
        if m:
            entity = m.group(1)
            phrase = m.group(2).lower().rstrip('.')
            pred = PREDICATE_ALIASES.get(phrase)
            if pred:
                return f"{pred}({entity})"

        # Fallback: look for known predicate phrase anywhere
        for phrase, pred in sorted(PREDICATE_ALIASES.items(),
                                   key=lambda x: -len(x[0])):
            if phrase in option_text.lower():
                # Try to find entity near the phrase
                m2 = re.search(r'([A-Z][a-z]+)', option_text)
                entity = m2.group(1) if m2 else "x"
                return f"{pred}({entity})"

        return None
