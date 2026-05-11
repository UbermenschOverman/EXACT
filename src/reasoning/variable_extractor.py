# src/reasoning/variable_extractor.py

import re
from typing import Dict, Tuple

class VariableExtractor:
    """
    Competition-grade variable extractor supporting extensive scientific notation,
    SI prefixes, unicode normalization, and canonical aliases.
    """

    # Value regex handles:
    # 10^-8, 10⁻⁸, 3 × 10^-6, 3x10^-6, 3e-6, 100, 100.5
    VALUE_REGEX = r'([\d.]*)\s*(?:[x×*eE]?\s*10\s*[\^]?\s*([⁻\-]?\d+)|[eE]([+-]?\d+)|[x×*]?\s*10\s*([⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹]+))?'

    PREFIXES = {
        'T': 1e12, 'tera': 1e12,
        'G': 1e9, 'giga': 1e9,
        'M': 1e6, 'mega': 1e6,
        'k': 1e3, 'kilo': 1e3,
        'c': 1e-2, 'centi': 1e-2,
        'm': 1e-3, 'milli': 1e-3,
        'μ': 1e-6, 'u': 1e-6, 'micro': 1e-6,
        'n': 1e-9, 'nano': 1e-9,
        'p': 1e-12, 'pico': 1e-12
    }
    
    SUPERSCRIPTS = {'⁻': '-', '⁺': '+', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}

    VARIABLE_ALIASES = {
        'U': 'V',
        'I': 'current',
        'q': 'charge',
        'Q': 'charge',
        'F_res': 'F',
        'E': 'energy',
        'C': 'capacitance'
    }

    UNIT_PATTERNS = [
        (r'V|volt|volts', 'V'),
        (r'A|amp|amps|ampere|amperes', 'A'),
        (r'Ω|ohm|ohms', 'Ω'),
        (r'W|watt|watts', 'W'),
        (r'F|farad|farads', 'F'),
        (r'J|joule|joules', 'J'),
        (r'C|coulomb|coulombs', 'C'),
        (r'N|newton|newtons', 'N'),
        (r'm|meter|meters', 'm'),
    ]

    def _parse_value(self, base: str, exp_standard: str, exp_e: str, exp_unicode: str) -> float:
        # If there's no base but there is an exponent (e.g., 10^-6), base is 1.0
        val = float(base) if base else 1.0
        exponent = 0
        
        if exp_standard:
            exponent = int(exp_standard)
        elif exp_e:
            exponent = int(exp_e)
        elif exp_unicode:
            norm_exp = "".join(self.SUPERSCRIPTS.get(c, c) for c in exp_unicode)
            exponent = int(norm_exp)
            
        return val * (10 ** exponent)

    def extract(self, text: str) -> Dict[str, float]:
        variables = {}
        
        for unit_pat, canon_unit in self.UNIT_PATTERNS:
            prefix_pat = r'([TGMkmμunp]?)'
            
            # The regex accounts for optional base value when standalone exponent is given like 10^-8
            full_regex = rf'([A-Za-z0-9_]+)\s*=\s*{self.VALUE_REGEX}\s*{prefix_pat}({unit_pat})\b'
            
            for match in re.finditer(full_regex, text):
                raw_var_name = match.group(1)
                base_val = match.group(2)
                exp_std = match.group(3)
                exp_e = match.group(4)
                exp_uni = match.group(5)
                prefix = match.group(6)
                
                # If everything is empty, this wasn't a valid match
                if not any([base_val, exp_std, exp_e, exp_uni]):
                    continue
                
                val = self._parse_value(base_val, exp_std, exp_e, exp_uni)
                
                if prefix and prefix in self.PREFIXES:
                    val *= self.PREFIXES[prefix]
                    
                # Apply canonical alias
                var_name = self.VARIABLE_ALIASES.get(raw_var_name, raw_var_name)
                    
                variables[var_name] = val

        return variables

    def extract_target(self, text: str) -> str:
        text_lower = text.lower()
        if "energy" in text_lower or "work" in text_lower: return self.VARIABLE_ALIASES.get("E", "E")
        if "force" in text_lower: return self.VARIABLE_ALIASES.get("F", "F")
        if "capacitance" in text_lower or "capacitor" in text_lower: return self.VARIABLE_ALIASES.get("C", "C")
        if "voltage" in text_lower or "potential" in text_lower: return self.VARIABLE_ALIASES.get("U", "U")
        if "current" in text_lower: return self.VARIABLE_ALIASES.get("I", "I")
        if "resistance" in text_lower: return self.VARIABLE_ALIASES.get("R", "R")
        if "charge" in text_lower: return self.VARIABLE_ALIASES.get("Q", "Q")
        
        return "Unknown"
