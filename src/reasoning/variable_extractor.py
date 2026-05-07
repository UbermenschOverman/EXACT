# src/reasoning/variable_extractor.py

"""
Hybrid regex + heuristic variable extractor.
Extracts variable names, values, and units from natural language physics questions.
"""

import re
from typing import Dict, Optional, Tuple, List


class VariableExtractor:
    """Extracts physics variables and their values from question text."""

    # Regex patterns: variable = value unit
    PATTERNS = [
        # V=10V, I=2A, R=5Ω etc.
        (r'([A-Za-z_]+)\s*=\s*([\d.]+)\s*(V|volt|volts)\b', "V"),
        (r'([A-Za-z_]+)\s*=\s*([\d.]+)\s*(A|amp|amps|ampere|amperes)\b', "A"),
        (r'([A-Za-z_]+)\s*=\s*([\d.]+)\s*(Ω|ohm|ohms)\b', "Ω"),
        (r'([A-Za-z_]+)\s*=\s*([\d.]+)\s*(W|watt|watts)\b', "W"),
        (r'([A-Za-z_]+)\s*=\s*([\d.]+)\s*(F|farad|farads)\b', "F"),
        (r'([A-Za-z_]+)\s*=\s*([\d.]+)\s*(J|joule|joules)\b', "J"),
        (r'([A-Za-z_]+)\s*=\s*([\d.]+)\s*(N|newton|newtons)\b', "N"),
        (r'([A-Za-z_]+)\s*=\s*([\d.]+)\s*(C|coulomb|coulombs)\b', "C"),
        (r'([A-Za-z_]+)\s*=\s*([\d.]+)\s*(N/C)\b', "N/C"),

        # Standalone values with units: "10V", "5 ohm", "2 amperes"
        (r'([\d.]+)\s*(V|volt|volts)\b', "V"),
        (r'([\d.]+)\s*(A|amp|amps|ampere|amperes)\b', "A"),
        (r'([\d.]+)\s*(Ω|ohm|ohms)\b', "Ω"),
        (r'([\d.]+)\s*(W|watt|watts)\b', "W"),
        (r'([\d.]+)\s*(F|farad|farads)\b', "F"),
        (r'([\d.]+)\s*(J|joule|joules)\b', "J"),
        (r'([\d.]+)\s*(N|newton|newtons)\b', "N"),
        (r'([\d.]+)\s*(C|coulomb|coulombs)\b', "C"),
    ]

    # Map unit strings to canonical variable names
    UNIT_TO_VAR = {
        "V": "V", "volt": "V", "volts": "V",
        "A": "I", "amp": "I", "amps": "I", "ampere": "I", "amperes": "I",
        "Ω": "R", "ohm": "R", "ohms": "R",
        "W": "P", "watt": "P", "watts": "P",
        "F": "C", "farad": "C", "farads": "C",
        "J": "E", "joule": "E", "joules": "E",
        "N": "F", "newton": "N", "newtons": "N",
        "C": "q", "coulomb": "q", "coulombs": "q",
        "N/C": "E_f",
    }

    # Target detection keywords
    TARGET_KEYWORDS = {
        "current": "I",
        "voltage": "V",
        "resistance": "R",
        "power": "P",
        "energy": "E",
        "capacitance": "C",
        "force": "F",
        "charge": "q",
        "electric field": "E_f",
        "total resistance": "R_total",
    }

    def extract(self, text: str) -> Dict[str, float]:
        """
        Extract variables and their numeric values from text.
        Returns dict like {"V": 10.0, "R": 5.0}.
        """
        variables = {}

        for pattern_tuple in self.PATTERNS:
            if len(pattern_tuple) == 2:
                pattern, canonical_unit = pattern_tuple[0], pattern_tuple[1]
            else:
                continue

            for match in re.finditer(pattern_tuple[0], text, re.IGNORECASE):
                groups = match.groups()

                if len(groups) == 3:
                    # variable = value unit
                    var_name = groups[0].strip()
                    value = float(groups[1])
                    unit = groups[2].strip()
                elif len(groups) == 2:
                    # value unit (standalone)
                    value = float(groups[0])
                    unit = groups[1].strip()
                    # Infer variable name from unit
                    var_name = self.UNIT_TO_VAR.get(unit, unit[0].upper())
                else:
                    continue

                # Map explicit variable names
                canonical = self._canonicalize_var(var_name, unit)
                variables[canonical] = value

        return variables

    def extract_target(self, text: str) -> Optional[str]:
        """
        Determine what variable to solve for based on question text.
        Looks for "calculate X", "find X", "what is X" patterns.
        """
        text_lower = text.lower()

        # Direct target patterns
        target_patterns = [
            r'(?:calculate|find|determine|compute|what is(?: the)?)\s+(?:the\s+)?(\w[\w\s]*?)(?:\s+if|\s+when|\s+given|\?|$)',
            r'(?:how much|how many)\s+(\w[\w\s]*?)(?:\s+if|\s+when|\s+given|\?|$)',
        ]

        for pattern in target_patterns:
            match = re.search(pattern, text_lower)
            if match:
                target_text = match.group(1).strip()
                # Map to variable name
                for keyword, var in self.TARGET_KEYWORDS.items():
                    if keyword in target_text:
                        return var

        # Fallback: find the variable NOT mentioned with a value
        mentioned_vars = set(self.extract(text).keys())
        for keyword, var in self.TARGET_KEYWORDS.items():
            if keyword in text_lower and var not in mentioned_vars:
                return var

        return None

    def _canonicalize_var(self, var_name: str, unit: str) -> str:
        """Map variable name to canonical form."""
        # First try explicit variable name
        var_upper = var_name.upper()
        if var_upper in ("V", "I", "R", "P", "E", "F", "C"):
            return var_upper
        if var_name.lower() in ("q", "e_f", "r_total", "r1", "r2", "r3"):
            return var_name

        # Fall back to unit-based mapping
        return self.UNIT_TO_VAR.get(unit, self.UNIT_TO_VAR.get(unit.lower(), var_name))
