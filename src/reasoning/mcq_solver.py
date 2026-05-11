# src/reasoning/mcq_solver.py

from typing import List, Dict, Any, Tuple
from src.reasoning.logic_ir import LogicProblem, MCQOption
from src.reasoning.premise_graph import PremiseGraph
from src.reasoning.parser import FOLParser

class MCQSolver:
    """
    Evaluates MCQ options deterministically.
    Scores based on derivability, contradiction checks, and proof depth.
    """
    def __init__(self):
        self.fol_parser = FOLParser()

    def evaluate(self, problem: LogicProblem) -> Tuple[MCQOption, Dict[str, float]]:
        if not problem.task or not problem.task.options:
            return None, {}
            
        best_option = None
        best_score = -float('inf')
        scores = {}
        
        # Build base premise graph once
        base_graph = PremiseGraph()
        for premise in problem.premises:
            if premise.ast:
                base_graph.add_premise(premise.id, premise.ast, is_given=True)
                
        # Forward chain to build full derivable space
        base_graph.forward_chain()
        base_contradiction = base_graph.detect_contradiction()
        
        for option in problem.task.options:
            score = 0.0
            
            # Since options in the dataset are NL and we don't have an LLM here,
            # we fallback to a naive heuristic for this demo.
            # In a full neuro-symbolic pipeline, `option.ast` would be populated by the LLM.
            if not option.ast:
                # Attempt to parse directly (works if the dataset provided FOL for options, but it doesn't)
                ast = self.fol_parser.parse_ast(option.text)
            else:
                ast = option.ast
                
            if not ast:
                scores[option.option_id] = -1.0
                continue
                
            ast_str = str(ast)
            
            # Check derivability (is it in the graph?)
            found_yes = any(node.text == ast_str for node in base_graph.premises.values())
            
            from src.reasoning.fol import Negation
            if isinstance(ast, Negation):
                neg_str = str(ast.statement)
            else:
                neg_str = str(Negation(ast))
                
            found_no = any(node.text == neg_str for node in base_graph.premises.values())
            
            if found_yes:
                score += 10.0
                # Penalize by depth (simulate proof depth by looking at derived_from tree if we traced it)
                # In PremiseGraph, derived nodes have IDs > 100
                target_node = next((n for n in base_graph.premises.values() if n.text == ast_str), None)
                if target_node and not target_node.is_given:
                    score -= 1.0 # Slight penalty for multi-hop vs direct given
            elif found_no:
                score -= 10.0 # Contradiction
                
            if base_contradiction:
                # Base system is inconsistent, ex falso quodlibet
                score = 0.0
                
            scores[option.option_id] = score
            
            if score > best_score:
                best_score = score
                best_option = option
                
        return best_option, scores
