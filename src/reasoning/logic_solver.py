# src/reasoning/logic_solver.py

import logging
from typing import Dict, List, Any
from src.reasoning.parser import FOLParser
from src.reasoning.premise_graph import PremiseGraph
from src.data.schema import ReasoningOutput

logger = logging.getLogger(__name__)

class LogicSolver:
    def __init__(self):
        self.parser = FOLParser()

    def solve(self, premises: List[str], target: str) -> ReasoningOutput:
        """
        Solve a logic query given FOL premises and a target question string.
        Returns ReasoningOutput.
        """
        graph = PremiseGraph()
        trace = []

        # 1. Parse and add given premises
        for i, p_str in enumerate(premises):
            ast = self.parser.parse_ast(p_str)
            node = graph.add_premise(f"P{i+1}", ast, is_given=True)
            trace.append({
                "step": len(trace) + 1,
                "action": "load_premise",
                "detail": f"Loaded premise: {node.text}",
                "premise_id": node.premise_id
            })

        # 2. Forward chain
        trace.append({
            "step": len(trace) + 1,
            "action": "forward_chain",
            "detail": "Starting multi-hop forward chaining"
        })
        
        derived_ids = graph.forward_chain()
        for d_id in derived_ids:
            node = graph.premises[d_id]
            trace.append({
                "step": len(trace) + 1,
                "action": "derive",
                "detail": f"Derived {node.text} from {node.derived_from}",
                "premise_id": node.premise_id
            })

        # 3. Check for contradictions
        contradiction = graph.detect_contradiction()
        if contradiction:
            trace.append({
                "step": len(trace) + 1,
                "action": "contradiction_check",
                "detail": contradiction
            })
            # Contradiction means the system is inconsistent, but for our purposes, we can return "No" or "Contradiction"
            return ReasoningOutput(
                question=target,
                answer="No",
                explanation=contradiction,
                reasoning_trace=trace,
                used_premises=graph.get_used_premises(),
                confidence=1.0,
                valid=True
            )

        # 4. Check if target is in the graph
        target_ast = self.parser.parse_ast(target)
        target_str = str(target_ast)
        
        # Determine if target or negated target is found
        found_yes = any(node.text == target_str for node in graph.premises.values())
        
        from src.reasoning.fol import Negation
        if isinstance(target_ast, Negation):
            neg_target_str = str(target_ast.statement)
        else:
            neg_target_str = str(Negation(target_ast))
            
        found_no = any(node.text == neg_target_str for node in graph.premises.values())

        if found_yes:
            answer = "Yes"
            confidence = 1.0
            explanation = f"Successfully derived {target_str}."
        elif found_no:
            answer = "No"
            confidence = 1.0
            explanation = f"Successfully derived the negation: {neg_target_str}."
        else:
            answer = "Unknown"
            confidence = 0.5
            explanation = f"Exhausted forward chaining. Could not derive {target_str} or its negation."

        trace.append({
            "step": len(trace) + 1,
            "action": "conclusion",
            "detail": explanation,
            "answer": answer
        })

        return ReasoningOutput(
            question=target,
            answer=answer,
            explanation=explanation,
            reasoning_trace=trace,
            used_premises=graph.get_used_premises(),
            confidence=confidence,
            valid=True
        )
