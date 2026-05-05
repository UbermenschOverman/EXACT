# src/reasoning/parser.py

import re


class FOLParser:
    def parse(self, fol_str: str):
        """
        Very basic parser (upgrade later)
        """
        fol_str = fol_str.strip()

        # detect implication
        if "->" in fol_str:
            parts = fol_str.split("->")
            return {
                "type": "implication",
                "premise": parts[0].strip(),
                "conclusion": parts[1].strip()
            }

        return {
            "type": "atomic",
            "value": fol_str
        }