# src/reasoning/parser.py

"""
FOL parser — extracts premises, rules, and implications from text.
Upgraded from basic string splitting to structured premise parsing.
"""

import re
from typing import List, Dict, Optional


class FOLParser:
    """Parser for First-Order Logic expressions and natural language premises."""

    # Patterns for detecting logical structures
    IMPLICATION_PATTERNS = [
        r"(.+?)\s*->\s*(.+)",           # A -> B
        r"(.+?)\s*→\s*(.+)",            # A → B
        r"if\s+(.+?)\s*,?\s*then\s+(.+)",  # if A, then B
        r"(.+?)\s+implies\s+(.+)",       # A implies B
    ]

    NEGATION_PATTERNS = [
        r"^not\s+(.+)",                  # not A
        r"^¬\s*(.+)",                    # ¬A
        r"^it is not the case that\s+(.+)",
    ]

    CONJUNCTION_PATTERNS = [
        r"(.+?)\s+and\s+(.+)",           # A and B
        r"(.+?)\s*∧\s*(.+)",            # A ∧ B
    ]

    def parse(self, fol_str: str) -> dict:
        """
        Parse a FOL string into structured representation.
        Enhanced version of original parser.
        """
        fol_str = fol_str.strip()

        # Try implication
        for pattern in self.IMPLICATION_PATTERNS:
            match = re.match(pattern, fol_str, re.IGNORECASE)
            if match:
                return {
                    "type": "implication",
                    "premise": match.group(1).strip(),
                    "conclusion": match.group(2).strip()
                }

        # Try negation
        for pattern in self.NEGATION_PATTERNS:
            match = re.match(pattern, fol_str, re.IGNORECASE)
            if match:
                return {
                    "type": "negation",
                    "value": match.group(1).strip()
                }

        # Try conjunction
        for pattern in self.CONJUNCTION_PATTERNS:
            match = re.match(pattern, fol_str, re.IGNORECASE)
            if match:
                return {
                    "type": "conjunction",
                    "left": match.group(1).strip(),
                    "right": match.group(2).strip()
                }

        return {
            "type": "atomic",
            "value": fol_str
        }

    def parse_premises(self, raw: str) -> List[Dict]:
        """
        Extract individual premises from a block of text.
        Handles numbered lists, bullet points, and line-separated premises.
        """
        premises = []
        
        # Split on newlines, numbered items, or bullet points
        lines = re.split(r'\n|(?:^|\n)\s*\d+[.)]\s*|(?:^|\n)\s*[-•]\s*', raw)
        
        pid_counter = 1
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parsed = self.parse(line)
            negation = parsed.get("type") == "negation"

            premises.append({
                "id": f"P{pid_counter}",
                "text": line,
                "negation": negation,
                "parsed": parsed,
            })
            pid_counter += 1

        return premises

    def parse_rules(self, premises: List[Dict]) -> List[Dict]:
        """
        Extract entailment rules from parsed premises.
        Implications become rules: premise -> conclusion.
        """
        rules = []

        for p in premises:
            parsed = p.get("parsed", {})

            if parsed.get("type") == "implication":
                # Find or create premise IDs for condition and conclusion
                condition_id = p["id"]
                conclusion_id = f"{p['id']}_conclusion"

                rules.append({
                    "id": f"R{len(rules) + 1}",
                    "conditions": [condition_id],
                    "conclusion": conclusion_id,
                    "text": f"If {parsed['premise']}, then {parsed['conclusion']}",
                    "condition_text": parsed["premise"],
                    "conclusion_text": parsed["conclusion"],
                })

        return rules

    def extract_premises_and_rules(self, raw: str) -> Dict:
        """
        Combined extraction: parse text into premises + rules.
        Returns structured data ready for LogicSolver.
        """
        premises = self.parse_premises(raw)
        rules = self.parse_rules(premises)

        return {
            "premises": premises,
            "rules": rules,
        }