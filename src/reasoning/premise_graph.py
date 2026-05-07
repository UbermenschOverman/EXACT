# src/reasoning/premise_graph.py

"""
Directed graph of premises with entailment edges.
Supports rule chaining, contradiction detection, and premise tracking.
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import deque


class PremiseNode:
    """A single premise in the graph."""

    def __init__(self, premise_id: str, text: str, is_given: bool = True):
        self.premise_id = premise_id
        self.text = text
        self.is_given = is_given       # True = from input, False = derived
        self.negation = False          # True if this is a negated premise
        self.derived_from: List[str] = []  # which premises led to this one


class Rule:
    """An entailment rule: if all conditions hold, then conclusion follows."""

    def __init__(self, rule_id: str, conditions: List[str], conclusion: str,
                 rule_text: str = ""):
        self.rule_id = rule_id
        self.conditions = conditions   # list of premise IDs
        self.conclusion = conclusion   # premise ID of conclusion
        self.rule_text = rule_text     # human-readable rule


class PremiseGraph:
    """Directed graph for managing premises and entailment relationships."""

    def __init__(self):
        self.premises: Dict[str, PremiseNode] = {}
        self.rules: List[Rule] = []
        self.edges: Dict[str, List[str]] = {}  # premise_id -> [reachable premise_ids]
        self._used_premises: Set[str] = set()

    def add_premise(self, premise_id: str, text: str, is_given: bool = True,
                    negation: bool = False) -> PremiseNode:
        """Add a premise to the graph."""
        node = PremiseNode(premise_id, text, is_given)
        node.negation = negation
        self.premises[premise_id] = node
        if premise_id not in self.edges:
            self.edges[premise_id] = []
        return node

    def add_rule(self, rule_id: str, conditions: List[str],
                 conclusion: str, rule_text: str = "") -> Rule:
        """Add an entailment rule: conditions => conclusion."""
        rule = Rule(rule_id, conditions, conclusion, rule_text)
        self.rules.append(rule)

        # Add edges from each condition to conclusion
        for cond in conditions:
            if cond in self.edges:
                self.edges[cond].append(conclusion)
            else:
                self.edges[cond] = [conclusion]

        return rule

    def forward_chain(self) -> List[str]:
        """
        Apply all rules via forward chaining.
        Returns list of newly derived premise IDs.
        """
        derived = []
        changed = True

        while changed:
            changed = False
            for rule in self.rules:
                # Check if all conditions are satisfied
                all_satisfied = all(
                    cid in self.premises for cid in rule.conditions
                )

                if all_satisfied and rule.conclusion not in self.premises:
                    # Derive new premise
                    conclusion_text = rule.rule_text or f"Derived from {rule.conditions}"
                    node = self.add_premise(
                        rule.conclusion,
                        conclusion_text,
                        is_given=False
                    )
                    node.derived_from = list(rule.conditions)
                    derived.append(rule.conclusion)
                    changed = True

                    # Track used premises
                    for cid in rule.conditions:
                        self._used_premises.add(cid)

        return derived

    def get_entailment_chain(self, target: str) -> List[str]:
        """
        BFS backward from target to find the chain of premises
        that lead to the target conclusion.
        """
        if target not in self.premises:
            return []

        chain = []
        visited = set()
        queue = deque([target])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            node = self.premises.get(current)
            if node:
                chain.append(current)
                self._used_premises.add(current)

                # Trace back through derivations
                for parent_id in node.derived_from:
                    if parent_id not in visited:
                        queue.append(parent_id)

        chain.reverse()
        return chain

    def detect_contradiction(self) -> Optional[str]:
        """
        Check if any premise and its negation both exist.
        Returns description of contradiction, or None.
        """
        premise_texts = {}

        for pid, node in self.premises.items():
            base_text = node.text.strip().lower()

            # Simple negation detection
            if base_text.startswith("not "):
                positive = base_text[4:]
                if positive in premise_texts:
                    return (
                        f"Contradiction: '{premise_texts[positive]}' and "
                        f"'{node.text}' (premises {premise_texts[positive]}, {pid})"
                    )
            else:
                negated = f"not {base_text}"
                for other_pid, other_node in self.premises.items():
                    if other_node.text.strip().lower() == negated:
                        return (
                            f"Contradiction: '{node.text}' and "
                            f"'{other_node.text}' (premises {pid}, {other_pid})"
                        )

            premise_texts[base_text] = pid

        return None

    def get_used_premises(self) -> List[str]:
        """Return list of premise texts that were used in reasoning."""
        used = []
        for pid in self._used_premises:
            if pid in self.premises:
                used.append(self.premises[pid].text)
        return used

    def get_all_premises(self) -> List[dict]:
        """Return all premises as dicts."""
        return [
            {
                "id": pid,
                "text": node.text,
                "is_given": node.is_given,
                "negation": node.negation,
                "derived_from": node.derived_from,
            }
            for pid, node in self.premises.items()
        ]
