# src/reasoning/narrative_extractor.py
"""
NarrativeExtractor — token-window based quantity extraction.

Handles patterns that the `= ` extractor misses:
  "voltage is 120 V"
  "current of 2 A"
  "a resistance of 5 ohms"
  "forces each 5 N"
  "two charges of 3e-8 C and 5e-8 C"
  "objects separated by 10 cm"
  "distance between them is 20 cm"

Returns structured QuantityHit objects with confidence scores.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantityHit:
    name: str                # canonical variable name (q1, V, current, ...)
    value: float
    unit: str
    confidence: float        # 0.0 – 1.0
    source_pattern: str      # which pattern triggered
    span: Tuple[int, int]    # character offsets in original text


# ─────────────────────────────────────────────────────────────────────────────
# Noun → physics variable mapping
# ─────────────────────────────────────────────────────────────────────────────

NOUN_TO_VAR: Dict[str, str] = {
    # Voltage
    "voltage": "V", "potential": "V", "emf": "V", "potential difference": "V",
    "electric potential": "V",
    # Current
    "current": "current", "amperage": "current",
    # Resistance
    "resistance": "R", "resistor": "R", "impedance": "R",
    # Power
    "power": "P", "wattage": "P",
    # Energy
    "energy": "energy", "work": "energy", "stored energy": "energy",
    # Capacitance
    "capacitance": "capacitance", "capacitor": "capacitance",
    # Force
    "force": "F", "electric force": "F", "net force": "F",
    "resultant force": "F", "resultant": "F",
    # Charge
    "charge": "q", "electric charge": "q",
    # Distance
    "distance": "r", "separation": "r", "radius": "r", "length": "r",
    # Angle
    "angle": "theta", "inclination": "theta",
    # Mass
    "mass": "m",
    # Acceleration
    "acceleration": "a",
}

# ─────────────────────────────────────────────────────────────────────────────
# SI prefixes
# ─────────────────────────────────────────────────────────────────────────────

SI_PREFIXES: Dict[str, float] = {
    "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3,
    "c": 1e-2, "m": 1e-3,
    "μ": 1e-6, "u": 1e-6, "n": 1e-9, "p": 1e-12,
}

UNIT_CANON: Dict[str, str] = {
    "V": "V", "volt": "V", "volts": "V",
    "A": "A", "amp": "A", "amps": "A", "ampere": "A", "amperes": "A",
    "Ω": "Ω", "ohm": "Ω", "ohms": "Ω",
    "W": "W", "watt": "W", "watts": "W",
    "F": "F", "farad": "F", "farads": "F",
    "J": "J", "joule": "J", "joules": "J",
    "C": "C", "coulomb": "C", "coulombs": "C",
    "N": "N", "newton": "N", "newtons": "N",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "cm": "cm", "mm": "mm", "km": "km",
    "kg": "kg", "g": "g",
    "rad": "rad", "°": "°", "deg": "°", "degree": "°", "degrees": "°",
}

# Conversion to SI base
TO_SI: Dict[str, float] = {
    "V": 1, "A": 1, "Ω": 1, "W": 1, "F": 1, "J": 1,
    "C": 1, "N": 1, "m": 1, "kg": 1, "rad": 1,
    "cm": 1e-2, "mm": 1e-3, "km": 1e3, "g": 1e-3,
    "°": 0.017453292519943295,  # pi/180
}

# ─────────────────────────────────────────────────────────────────────────────
# Number pattern
# ─────────────────────────────────────────────────────────────────────────────

_SUPERS = {'⁻': '-', '⁺': '+', '⁰': '0', '¹': '1', '²': '2', '³': '3',
           '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}

_NUM = (
    r'([-+]?\d+(?:\.\d+)?)'           # base
    r'(?:'
    r'\s*[×x\*·]\s*10\s*\^\s*([-+]?\d+)'   # × 10^exp
    r'|[eE]([-+]?\d+)'                       # e-notation
    r')?'
)

_PREFIX = r'([TGMkmμuncp]?)'
_UNIT   = r'([A-Za-zΩ°μ]+)'


def _parse_num(base: str, exp_hat: Optional[str], exp_e: Optional[str]) -> float:
    val = float(base) if base else 1.0
    exp = 0
    if exp_hat:
        exp = int(exp_hat)
    elif exp_e:
        exp = int(exp_e)
    return val * (10 ** exp)


def _canon_unit(prefix: str, raw_unit: str) -> Tuple[str, float]:
    """Return (canonical_unit_string, multiplier_to_SI)."""
    raw = raw_unit.strip()
    # Check prefix+unit combo (e.g. "μF", "cm")
    combo = prefix + raw
    if combo in UNIT_CANON:
        cu = UNIT_CANON[combo]
        mult = TO_SI.get(cu, 1.0) * SI_PREFIXES.get(prefix, 1.0)
        return cu, mult
    # Without prefix
    if raw in UNIT_CANON:
        cu = UNIT_CANON[raw]
        mult = TO_SI.get(cu, 1.0) * SI_PREFIXES.get(prefix, 1.0)
        return cu, mult
    return raw, SI_PREFIXES.get(prefix, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

class NarrativeExtractor:
    """
    Token-window extraction for narrative physics problems.
    Does NOT require Var = Value pattern.
    """

    # Noun → (variable_template, allow_plural)
    # variable_template: "q{n}" for multi-entity, plain for single
    _NOUN_PATTERNS = [
        # "voltage is 120 V", "voltage of 120 V"
        (r'(?:voltage|potential(?:\s+difference)?|emf)\s+(?:is|of|=|:)\s*', "V", False),
        # "charged to 200 V", "charged with 30 V"
        (r'charged\s+(?:to|with|at)\s*', "V", False),
        # "current of 2 A", "current is 2 A"
        (r'current\s+(?:of|is|=)\s*', "current", False),
        # "resistance of 5 Ω", "resistance 5 Ω" (bare noun)
        (r'(?:resistance|resistor|impedance)\s+(?:of\s+|is\s+|=\s*)?', "R", False),
        # "power of 600 W"
        (r'power\s+(?:of|is|=)\s*', "P", False),
        # "energy of 45 J", "energy stored is 45 J"
        (r'(?:energy|work)\s+(?:of|is|=|stored\s+(?:is\s+)?)\s*', "energy", False),
        # "capacitor of 50 μF", "capacitance 50 μF" (bare noun)
        (r'(?:capacitor|capacitance)\s+(?:of\s+|is\s+|=\s*)?', "capacitance", False),
        # "force of 5 N", "forces each 5 N", "force 5 N" (bare)
        (r'(?:electric\s+)?forces?\s+(?:of\s+|each\s+|is\s+|=\s*)?', "F", True),
        # "magnitude of 5 N"
        (r'magnitude\s+(?:of\s+|is\s+|=\s*)', "F", False),
        # "charge of 3e-8 C"
        (r'(?:electric\s+)?charges?\s+(?:of\s+|is\s+|=\s*)?', "q", True),
        # angle
        (r'(?:at\s+(?:an\s+)?)?angle\s+(?:of\s+|is\s+|=\s*)?', "theta", False),
    ]

    # Distance narrative patterns
    _DISTANCE_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*(cm|mm|m|km)\s+apart',
        r'separated\s+by\s+(\d+(?:\.\d+)?)\s*(cm|mm|m|km)',
        r'distance\s+(?:between\s+\w+\s+is|of|=|is)\s+(\d+(?:\.\d+)?)\s*(cm|mm|m|km)',
        r'(\d+(?:\.\d+)?)\s*(cm|mm|m|km)\s+(?:from\s+each\s+other|away)',
    ]

    def extract(self, text: str) -> List[QuantityHit]:
        hits: List[QuantityHit] = []

        # ── 1. Noun-triggered extraction ─────────────────────────────
        for noun_pat, var_name, allow_plural in self._NOUN_PATTERNS:
            full_pat = noun_pat + _NUM + r'\s*' + _PREFIX + _UNIT + r'\b'
            for m in re.finditer(full_pat, text, re.IGNORECASE):
                try:
                    val = _parse_num(m.group(1), m.group(2), m.group(3))
                    prefix = m.group(4) or ""
                    raw_unit = m.group(5)
                    cu, mult = _canon_unit(prefix, raw_unit)
                    si_val = val * mult

                    hits.append(QuantityHit(
                        name=var_name,
                        value=si_val,
                        unit=cu,
                        confidence=0.85,
                        source_pattern=noun_pat,
                        span=(m.start(), m.end()),
                    ))
                except (ValueError, IndexError):
                    continue

        # ── 2. "X and Y" multi-value extraction for charges/forces ───
        # "charges of 3e-8 C and 5e-8 C"
        multi_pat = (
            r'(?:charges?|forces?)\s+of\s+'
            + _NUM + r'\s*' + _PREFIX + _UNIT + r'\s+and\s+'
            + _NUM + r'\s*' + _PREFIX + _UNIT + r'\b'
        )
        for m in re.finditer(multi_pat, text, re.IGNORECASE):
            try:
                v1 = _parse_num(m.group(1), m.group(2), m.group(3))
                p1, u1 = m.group(4) or "", m.group(5)
                cu1, mult1 = _canon_unit(p1, u1)

                v2 = _parse_num(m.group(6), m.group(7), m.group(8))
                p2, u2 = m.group(9) or "", m.group(10)
                cu2, mult2 = _canon_unit(p2, u2)

                noun = m.group(0).split()[0].lower().rstrip('s')
                base = "q" if "charg" in noun else "F"

                hits.append(QuantityHit(base + "1", v1 * mult1, cu1, 0.82, "multi_pair", (m.start(), m.end())))
                hits.append(QuantityHit(base + "2", v2 * mult2, cu2, 0.82, "multi_pair", (m.start(), m.end())))
            except (ValueError, IndexError):
                continue

        # ── 3. "each X N" — homogeneous plural ─────────────────────
        each_pat = r'each\s+(?:with\s+(?:a\s+)?(?:magnitude\s+of\s+)?)?'  + _NUM + r'\s*' + _PREFIX + _UNIT + r'\b'
        counter: Dict[str, int] = {}
        for m in re.finditer(each_pat, text, re.IGNORECASE):
            try:
                val = _parse_num(m.group(1), m.group(2), m.group(3))
                prefix = m.group(4) or ""
                raw_unit = m.group(5)
                cu, mult = _canon_unit(prefix, raw_unit)
                si_val = val * mult

                # Determine noun from surrounding context
                context = text[max(0, m.start()-40):m.start()].lower()
                if "force" in context or cu == "N":
                    base = "F"
                elif "charge" in context or cu == "C":
                    base = "q"
                else:
                    base = "x"

                counter[base] = counter.get(base, 0) + 1
                name = f"{base}{counter[base]}"
                hits.append(QuantityHit(name, si_val, cu, 0.75, "each_pattern", (m.start(), m.end())))
                # Add second copy
                counter[base] += 1
                hits.append(QuantityHit(f"{base}{counter[base]}", si_val, cu, 0.75, "each_pattern", (m.start(), m.end())))
            except (ValueError, IndexError):
                continue

        # ── 4. Distance narrative patterns ──────────────────────────
        for pat in self._DISTANCE_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                try:
                    val = float(m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0).split()[0])
                    unit = m.group(2) if m.lastindex and m.lastindex >= 2 else "m"
                    cu, mult = _canon_unit("", unit)
                    si_val = val * mult
                    hits.append(QuantityHit("r", si_val, "m", 0.88, "distance_narrative", (m.start(), m.end())))
                except (ValueError, IndexError):
                    continue

        return self._deduplicate(hits)

    def _deduplicate(self, hits: List[QuantityHit]) -> List[QuantityHit]:
        """Remove lower-confidence hits that duplicate the same variable name."""
        seen: Dict[str, QuantityHit] = {}
        for h in sorted(hits, key=lambda x: -x.confidence):
            if h.name not in seen:
                seen[h.name] = h
        return list(seen.values())

    def to_dict(self, hits: List[QuantityHit]) -> Dict[str, float]:
        """Convert hits to {name: value} dict for formula bank."""
        return {h.name: h.value for h in hits}
