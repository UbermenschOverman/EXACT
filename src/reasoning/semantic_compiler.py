# src/reasoning/semantic_compiler.py
"""
SemanticCompiler — the central grounding layer for EXACT.

Pipeline:
  RAW TEXT / DICT
    ↓ Layer A1: Structural Extraction (explicit Var = Value Unit)
    ↓ Layer A2: Narrative Extraction (token-window, noun→quantity)
    ↓ Layer B:  Ontology Normalization
    ↓ Layer C:  LLM Compiler (optional, OFF by default)
  WorldModel

The SemanticCompiler is the ONLY component allowed to touch raw text.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.reasoning.world_model import WorldModel, Quantity, MCQOption, Goal
from src.reasoning.ontology import (
    SI_PREFIXES, PREDICATE_ALIASES, question_to_fol_target,
    canonicalize_predicate_phrase, normalize_variable, CANONICAL_UNITS,
)
from src.reasoning.physics_scene_builder import PhysicsSceneBuilder
from src.reasoning.question_translator import QuestionTranslator
from src.reasoning.narrative_extractor import NarrativeExtractor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER A1 — Structural Extraction (explicit Var = Value Unit)
# ─────────────────────────────────────────────────────────────────────────────

class StructuralExtractor:
    """Handles explicit format: q1 = 6e-8 C, C = 100 μF, U = 30 V."""

    _SUPERS = {'⁻': '-', '⁺': '+', '⁰': '0', '¹': '1', '²': '2', '³': '3',
               '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}

    _VALUE = (
        r'([-+]?\d+(?:\.\d+)?)'
        r'(?:'
        r'\s*[×x\*·]\s*10\s*[\^]\s*([-+]?\d+)'
        r'|[eE]([-+]?\d+)'
        r'|'
        r'\s*[×x\*·]\s*10\s*((?:[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))'
        r')?'
    )

    _UNIT_PATTERNS = [
        (r'C|coulomb(?:s)?',   'C'),
        (r'V|volt(?:s)?',      'V'),
        (r'A|amp(?:ere)?(?:s)?', 'A'),
        (r'Ω|ohm(?:s)?',       'Ω'),
        (r'W|watt(?:s)?',      'W'),
        (r'F|farad(?:s)?',     'F'),
        (r'J|joule(?:s)?',     'J'),
        (r'N|newton(?:s)?',    'N'),
    ]

    def _parse_value(self, base, exp_hat, exp_e, exp_uni):
        val = float(base) if base else 1.0
        exp = 0
        if exp_hat:
            exp = int(exp_hat)
        elif exp_e:
            exp = int(exp_e)
        elif exp_uni:
            exp = int(''.join(self._SUPERS.get(c, c) for c in exp_uni))
        return val * (10 ** exp)

    def extract_quantities(self, text: str) -> List[Tuple[str, float, str]]:
        results: List[Tuple[str, float, str]] = []
        for unit_pat, canon_unit in self._UNIT_PATTERNS:
            prefix_pat = r'([TGMkmμuncp]?)'
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
                results.append((raw_name, val, canon_unit))
        return results

    def extract_named_charges(self, text: str) -> List[Tuple[str, float, str]]:
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
# LAYER B — Ontology Normalization
# ─────────────────────────────────────────────────────────────────────────────

class OntologyNormalizer:
    _TARGET_MAP = [
        (["force", "resultant", "net force"], "F"),
        (["energy", "stored energy", "work"], "energy"),
        (["capacitance"], "capacitance"),
        (["voltage", "potential difference", "emf"], "V"),
        (["current"], "current"),
        (["resistance"], "R"),
        (["charge"], "charge"),
        (["power"], "P"),
    ]

    def detect_target(self, text: str) -> Optional[str]:
        t = text.lower()

        # Strategy 1: find the noun near a question keyword
        # "Find the X", "What is the X", "Calculate X", "Determine X"
        import re
        question_pats = [
            r'(?:find|calculate|determine|compute|what is)\s+(?:the\s+)?(\w[\w\s]*?)(?:\?|\.|$|,)',
        ]
        for pat in question_pats:
            m = re.search(pat, t)
            if m:
                target_phrase = m.group(1).strip()
                for keywords, var in self._TARGET_MAP:
                    if any(kw in target_phrase for kw in keywords):
                        return var

        # Strategy 2: fallback — check full text, but prioritize later sentences
        # Split on periods/question marks to find the question sentence
        sentences = re.split(r'[.?]', t)
        question_sentence = sentences[-1] if sentences else t
        for keywords, var in self._TARGET_MAP:
            if any(kw in question_sentence for kw in keywords):
                return var

        # Strategy 3: full-text scan
        for keywords, var in self._TARGET_MAP:
            if any(kw in t for kw in keywords):
                return var

        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SEMANTIC COMPILER
# ─────────────────────────────────────────────────────────────────────────────

class SemanticCompiler:
    """Central grounding layer. Single source of truth for text → WorldModel."""

    def __init__(self, llm_compiler=None):
        self.extractor = StructuralExtractor()
        self.narrative = NarrativeExtractor()
        self.normalizer = OntologyNormalizer()
        self.scene_builder = PhysicsSceneBuilder()
        self.translator = QuestionTranslator(llm_compiler)
        self.llm = llm_compiler

    # ── Physics compilation ────────────────────────────────────────────

    def compile_physics(self, text: str) -> WorldModel:
        wm = WorldModel(raw_text=text, problem_type="physics")

        # Layer A1 — explicit extraction (highest confidence)
        named_charges = self.extractor.extract_named_charges(text)
        all_quantities = self.extractor.extract_quantities(text)

        for name, val, unit in named_charges:
            wm.set_quantity(name, "value", val, unit, "explicit")

        charge_counter = 1
        for raw_name, val, unit in all_quantities:
            if raw_name in ("charge", "q") and named_charges:
                continue
            ent_name = raw_name if raw_name not in ("charge", "q") \
                       else f"q{charge_counter}"
            if raw_name in ("charge", "q"):
                charge_counter += 1
            wm.set_quantity(ent_name, "value", val, unit, "explicit")

        # Layer A2 — narrative extraction (fills gaps)
        narrative_hits = self.narrative.extract(text)
        for hit in narrative_hits:
            if hit.name not in wm.flat_quantities():
                wm.set_quantity(hit.name, "value", hit.value, hit.unit, "narrative")

        # Layer B — detect goal / target
        target = self.normalizer.detect_target(text)
        if target:
            wm.goals.append(Goal("compute", target))

        # Layer C — optional LLM enrichment
        if self.llm and not wm.entities:
            llm_out = self.llm.compile_physics(text)
            if llm_out:
                self._apply_llm_physics(wm, llm_out)
                wm.compilation_method = "llm"
            else:
                wm.compilation_method = "failed"
        else:
            wm.compilation_method = "structural" if wm.entities else "failed"

        wm.compilation_confidence = 1.0 if wm.entities else 0.0
        self.scene_builder.enrich(wm)
        return wm

    def _apply_llm_physics(self, wm: WorldModel, llm_out: Dict[str, Any]):
        for ent in llm_out.get("entities", []):
            val = ent.get("value")
            if val is not None:
                wm.set_quantity(ent.get("name", "?"), "value",
                               float(val), ent.get("unit"), "llm")
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
        wm = WorldModel(problem_type="logic")

        premises_fol: List[str] = item.get("premises-FOL", [])
        wm.premises_fol = premises_fol

        questions: List[str] = item.get("questions", [])
        if not questions:
            return wm

        q_text = questions[0]
        wm.raw_text = q_text

        # Stem-only translation to prevent option text triggering false matches
        q_stem = q_text.split("\n")[0].strip()
        trans = self.translator.translate(q_stem, premises_fol)

        if trans.task_type == "mcq":
            goal = Goal("mcq", "__mcq__")
            self._extract_mcq_options(q_text, goal)
            wm.goals.append(goal)

        elif trans.task_type == "entailment" and trans.fol_target:
            wm.goals.append(Goal("entailment", trans.fol_target))

        else:
            # Structural MCQ fallback: A./B./C. in full text
            if re.search(r'\b[A-D]\.', q_text):
                goal = Goal("mcq", "__mcq__")
                self._extract_mcq_options(q_text, goal)
                wm.goals.append(goal)
            # "according to the premises" entailment fallback
            elif re.search(r'according to the premises|does it follow', q_text, re.IGNORECASE):
                wm.goals.append(Goal("entailment", None))
            else:
                logger.debug(f"QuestionTranslator failed: {trans.source} / {trans.explanation}")

        method_parts = [trans.source]
        if wm.goals:
            method_parts.append("goal_found")
        wm.compilation_method = "+".join(method_parts)
        wm.compilation_confidence = trans.confidence
        return wm

    def _extract_mcq_options(self, q_text: str, goal: Goal):
        """Extract A./B./C./D. options from question text."""
        lines = q_text.splitlines()
        for line in lines:
            line = line.strip()
            m = re.match(r'^([A-D])\.\s*(.+)', line)
            if m:
                opt_id, opt_text = m.group(1), m.group(2).strip()
                fol_opt = self._canonicalize_option(opt_text)
                goal.options.append(MCQOption(opt_id, opt_text, fol_opt))
        # Inline fallback
        if not goal.options:
            for m in re.finditer(r'\b([A-D])\.\s*(.+?)(?=\s*[A-D]\.|$)', q_text, re.DOTALL):
                opt_id, opt_text = m.group(1), m.group(2).strip()
                fol_opt = self._canonicalize_option(opt_text)
                goal.options.append(MCQOption(opt_id, opt_text, fol_opt))

    def _canonicalize_option(self, option_text: str) -> Optional[str]:
        """NL MCQ option → FOL string via ontology + implication detection."""
        # 1. Implication: "If A, then B" / "If A then B"
        imp_match = re.match(
            r'[Ii]f\s+(?:a |an |all |the |every )?(.+?),?\s+then\s+(?:it\s+)?(?:is\s+|must\s+be\s+)?(.+)',
            option_text
        )
        if imp_match:
            antecedent = imp_match.group(1).strip().rstrip(',.')
            consequent = imp_match.group(2).strip().rstrip('.')
            pred_a = canonicalize_predicate_phrase(antecedent)
            pred_c = canonicalize_predicate_phrase(consequent)
            if pred_a and pred_c:
                return f"∀x ({pred_a}(x) → {pred_c}(x))"
            elif pred_a:
                return f"∀x ({pred_a}(x) → {consequent}(x))"
            elif pred_c:
                return f"∀x ({antecedent}(x) → {pred_c}(x))"
            return None

        # 2. Entity + predicate: "Sophia qualifies for scholarship"
        m = re.match(r'([A-Z][a-z]+)\s+(.+)', option_text)
        if m:
            entity = m.group(1)
            phrase = m.group(2).lower().rstrip('.')
            pred = canonicalize_predicate_phrase(phrase)
            if pred and not pred.startswith("¬"):
                return f"{pred}({entity})"

        # 3. Longest matching predicate phrase
        pred = canonicalize_predicate_phrase(option_text)
        if pred and not pred.startswith("¬"):
            m2 = re.search(r'([A-Z][a-z]+)', option_text)
            entity = m2.group(1) if m2 else "x"
            return f"{pred}({entity})"

        return None
