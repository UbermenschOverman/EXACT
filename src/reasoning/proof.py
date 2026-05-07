# src/reasoning/proof.py

"""
Proof trace generator — builds structured step-by-step proofs
from premise graphs and solver results.
"""

from typing import List, Dict, Any, Optional
from src.reasoning.premise_graph import PremiseGraph


class ProofGenerator:
    """Generates human-readable and structured proof traces."""

    def generate(self, reasoning_trace: List[Dict],
                 graph: Optional[PremiseGraph] = None) -> List[Dict]:
        """
        Generate structured proof from reasoning trace.

        Args:
            reasoning_trace: Raw trace from solver
            graph: Optional premise graph for richer output

        Returns:
            List of proof steps with human-readable descriptions.
        """
        proof_steps = []

        for entry in reasoning_trace:
            action = entry.get("action", "")

            if action == "add_premise":
                proof_steps.append({
                    "step": len(proof_steps) + 1,
                    "type": "given",
                    "description": f"Given: {entry.get('text', '')}",
                    "premise_id": entry.get("premise_id"),
                    "source": "input",
                })

            elif action == "add_rule":
                proof_steps.append({
                    "step": len(proof_steps) + 1,
                    "type": "rule",
                    "description": f"Rule: {entry.get('text', '')}",
                    "rule_id": entry.get("rule_id"),
                    "conditions": entry.get("conditions", []),
                    "conclusion": entry.get("conclusion"),
                })

            elif action == "derive":
                proof_steps.append({
                    "step": len(proof_steps) + 1,
                    "type": "derivation",
                    "description": (
                        f"By applying rule to {entry.get('derived_from', [])}, "
                        f"derive: {entry.get('text', '')}"
                    ),
                    "premise_id": entry.get("premise_id"),
                    "derived_from": entry.get("derived_from", []),
                })

            elif action == "detect_contradiction":
                proof_steps.append({
                    "step": len(proof_steps) + 1,
                    "type": "contradiction",
                    "description": f"⚠ {entry.get('detail', '')}",
                })

            elif action == "check_target":
                reached = entry.get("reached", False)
                proof_steps.append({
                    "step": len(proof_steps) + 1,
                    "type": "target_check",
                    "description": (
                        f"Target '{entry.get('target')}' "
                        f"{'reached via chain: ' + str(entry.get('chain', [])) if reached else 'NOT reachable'}"
                    ),
                    "reached": reached,
                })

            elif action == "conclude":
                proof_steps.append({
                    "step": len(proof_steps) + 1,
                    "type": "conclusion",
                    "description": f"Conclusion: {entry.get('answer', '')} (confidence: {entry.get('confidence', 0):.2f})",
                    "answer": entry.get("answer"),
                    "confidence": entry.get("confidence", 0),
                })

            elif action in ("identify_formula", "extract_variables",
                            "solve", "compute", "substitute"):
                # Physics solver steps — pass through
                proof_steps.append({
                    "step": len(proof_steps) + 1,
                    "type": action,
                    "description": entry.get("detail", entry.get("text", str(entry))),
                    **{k: v for k, v in entry.items() if k not in ("step", "action")},
                })

        return proof_steps

    def generate_legacy(self, parsed_fol, result) -> List[str]:
        """
        Legacy proof generation for backward compatibility.
        Returns list of strings (old format).
        """
        steps = []

        if isinstance(parsed_fol, dict):
            if parsed_fol.get("type") == "implication":
                steps.append(f"Given premise: {parsed_fol['premise']}")
                steps.append(f"If premise holds, then: {parsed_fol['conclusion']}")

        steps.append(f"Result: {result}")
        return steps