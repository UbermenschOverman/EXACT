# src/reasoning/logic_solver.py

"""
Logic solver using premise graph + forward chaining.
Replaces the stub SymPy-only solver for logic-type questions.
"""

from typing import List, Dict, Any, Optional
from src.reasoning.premise_graph import PremiseGraph


class LogicSolver:
    """
    Solves logic reasoning questions by:
    1. Building a premise graph from LLM-extracted premises
    2. Applying forward chaining rules
    3. Checking if target conclusion is reachable
    4. Generating proof traces with used premises
    """

    def solve(self, premises: List[Dict], rules: List[Dict],
              target: Optional[str] = None) -> Dict[str, Any]:
        """
        Main solve method.

        Args:
            premises: List of {"id": "P1", "text": "...", "negation": False}
            rules: List of {"id": "R1", "conditions": ["P1","P2"], "conclusion": "P3", "text": "..."}
            target: Optional target conclusion to check

        Returns:
            {
                "answer": str,
                "derived": List[str],
                "used_premises": List[str],
                "reasoning_trace": List[dict],
                "contradiction": Optional[str],
                "confidence": float,
                "graph_summary": dict
            }
        """
        graph = PremiseGraph()

        # 1. Add all premises
        trace = []
        for p in premises:
            pid = p.get("id", f"P{len(graph.premises)}")
            text = p.get("text", "")
            negation = p.get("negation", False)
            graph.add_premise(pid, text, is_given=True, negation=negation)
            trace.append({
                "step": len(trace) + 1,
                "action": "add_premise",
                "premise_id": pid,
                "text": text,
                "source": "given",
            })

        # 2. Add rules
        for r in rules:
            rid = r.get("id", f"R{len(graph.rules)}")
            conditions = r.get("conditions", [])
            conclusion = r.get("conclusion", "")
            rule_text = r.get("text", "")
            graph.add_rule(rid, conditions, conclusion, rule_text)
            trace.append({
                "step": len(trace) + 1,
                "action": "add_rule",
                "rule_id": rid,
                "conditions": conditions,
                "conclusion": conclusion,
                "text": rule_text,
            })

        # 3. Check for contradictions
        contradiction = graph.detect_contradiction()
        if contradiction:
            trace.append({
                "step": len(trace) + 1,
                "action": "detect_contradiction",
                "detail": contradiction,
            })

        # 4. Forward chain
        derived = graph.forward_chain()
        for d in derived:
            node = graph.premises.get(d)
            trace.append({
                "step": len(trace) + 1,
                "action": "derive",
                "premise_id": d,
                "text": node.text if node else "",
                "derived_from": node.derived_from if node else [],
            })

        # 5. Check target
        answer = "UNKNOWN"
        target_reached = False

        if target:
            if target in graph.premises:
                answer = graph.premises[target].text
                target_reached = True
                chain = graph.get_entailment_chain(target)
                trace.append({
                    "step": len(trace) + 1,
                    "action": "check_target",
                    "target": target,
                    "reached": True,
                    "chain": chain,
                })
            else:
                trace.append({
                    "step": len(trace) + 1,
                    "action": "check_target",
                    "target": target,
                    "reached": False,
                })
        elif derived:
            # No specific target — use last derived conclusion
            last = derived[-1]
            answer = graph.premises[last].text
            target_reached = True

        # 6. Compute confidence
        confidence = self._compute_confidence(
            graph, derived, contradiction, target_reached
        )

        trace.append({
            "step": len(trace) + 1,
            "action": "conclude",
            "answer": answer,
            "confidence": confidence,
        })

        return {
            "answer": answer,
            "derived": derived,
            "used_premises": graph.get_used_premises(),
            "reasoning_trace": trace,
            "contradiction": contradiction,
            "confidence": confidence,
            "graph_summary": {
                "total_premises": len(graph.premises),
                "given": sum(1 for p in graph.premises.values() if p.is_given),
                "derived": len(derived),
                "rules_applied": len(graph.rules),
            }
        }

    def _compute_confidence(self, graph: PremiseGraph,
                            derived: List[str],
                            contradiction: Optional[str],
                            target_reached: bool) -> float:
        """
        Heuristic confidence score based on:
        - Whether target was reached
        - Number of derivation steps
        - Presence of contradictions
        """
        if contradiction:
            return 0.1  # low confidence if contradictions exist

        if not target_reached and not derived:
            return 0.2  # no derivations = uncertain

        confidence = 0.5  # base

        # Bonus for reaching target
        if target_reached:
            confidence += 0.3

        # Bonus for having derivation chain
        if derived:
            confidence += min(0.2, len(derived) * 0.05)

        return min(confidence, 1.0)
