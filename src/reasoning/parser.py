# src/reasoning/parser.py

"""
FOL parser — extracts premises, rules, and implications from text.
Upgraded to parse Unicode FOL strings into formal AST representations.
"""

import re
from typing import List, Dict, Optional, Tuple
from src.reasoning.fol import (
    ASTNode, Variable, Constant, Predicate, 
    Negation, Conjunction, Implication, ForAll, Exists
)

class FOLParser:
    """Parser for First-Order Logic expressions and natural language premises."""

    def parse_ast(self, s: str) -> ASTNode:
        """Parse a string into an AST Node."""
        s = s.strip()
        
        # Normalize text aliases
        s = s.replace("forall", "∀")
        s = s.replace("->", "→")
        s = s.replace("⟹", "→")
        s = s.replace("implies", "→")

        # Handle Python-style ForAll(x, Body) and Exists(x, Body)
        # e.g. "ForAll(x, (A(x) → B(x)))" or "∀x (A(x) → B(x))"
        m_fa = re.match(r'ForAll\s*\(\s*([a-z])\s*,\s*(.+)\)\s*$', s, re.DOTALL)
        if m_fa:
            var, body = m_fa.group(1), m_fa.group(2).strip()
            return ForAll(Variable(var), self.parse_ast(body))

        m_ex = re.match(r'Exists\s*\(\s*([a-z])\s*,\s*(.+)\)\s*$', s, re.DOTALL)
        if m_ex:
            var, body = m_ex.group(1), m_ex.group(2).strip()
            return Exists(Variable(var), self.parse_ast(body))

        # Broken unicode: "∃forall" artifacts
        s = re.sub(r'[∃∀]forall', '∀', s)
        s = re.sub(r'[∃]exists', '∃', s)
        
        # Strip outer parens if they exist and are matched
        if s.startswith('(') and s.endswith(')'):
            # Check if outer parens match
            depth = 0
            matched = True
            for i, c in enumerate(s):
                if c == '(': depth += 1
                elif c == ')': depth -= 1
                if depth == 0 and i < len(s) - 1:
                    matched = False
                    break
            if matched:
                return self.parse_ast(s[1:-1])

        # ForAll: ∀x (Statement) or ∀x, Statement or ∀(x, Statement)
        match = re.match(r'∀\s*\(?\s*([a-z])\s*[,(]?\s*(.+)', s)
        if match:
            var_name, rest = match.groups()
            if rest.endswith(')'): rest = rest[:-1]
            return ForAll(Variable(var_name), self.parse_ast(rest))

        # Exists: ∃x (Statement) or ∃(x, Statement)
        match = re.match(r'∃\s*\(?\s*([a-z])\s*[,(]?\s*(.+)', s)
        if match:
            var_name, rest = match.groups()
            if rest.endswith(')'): rest = rest[:-1]
            return Exists(Variable(var_name), self.parse_ast(rest))

        # Implication: split by top-level →
        depth = 0
        for i, c in enumerate(s):
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            elif c == '→' and depth == 0:
                left = s[:i].strip()
                right = s[i+1:].strip()
                return Implication(self.parse_ast(left), self.parse_ast(right))

        # Conjunction: split by top-level ∧
        depth = 0
        for i, c in enumerate(s):
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            elif c == '∧' and depth == 0:
                left = s[:i].strip()
                right = s[i+1:].strip()
                return Conjunction(self.parse_ast(left), self.parse_ast(right))

        # Negation: ¬Statement
        if s.startswith('¬'):
            return Negation(self.parse_ast(s[1:].strip()))
            
        # Predicate: Name(arg1, arg2)
        match = re.match(r'([A-Za-z0-9_]+)\((.+)\)', s)
        if match:
            name, args_str = match.groups()
            args_list = [arg.strip() for arg in args_str.split(',')]
            ast_args = []
            for arg in args_list:
                if arg.islower():
                    ast_args.append(Variable(arg))
                else:
                    ast_args.append(Constant(arg))
            return Predicate(name, ast_args)

        # Fallback to atomic predicate with no args
        return Predicate(s, [])

    def parse_premises(self, raw: str) -> List[Dict]:
        """
        Extract individual premises from a block of text.
        """
        premises = []
        lines = re.split(r'\n|(?:^|\n)\s*\d+[.)]\s*|(?:^|\n)\s*[-•]\s*', raw)
        pid_counter = 1
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            try:
                ast = self.parse_ast(line)
                negation = isinstance(ast, Negation)
            except Exception:
                ast = None
                negation = False

            premises.append({
                "id": f"P{pid_counter}",
                "text": line,
                "negation": negation,
                "ast": ast,
            })
            pid_counter += 1

        return premises

    def extract_premises_and_rules(self, raw: str) -> Dict:
        """
        Extract premises (returns flat list).
        """
        premises = self.parse_premises(raw)
        return {
            "premises": premises,
            "rules": []  # rules are integrated into premise ASTs
        }