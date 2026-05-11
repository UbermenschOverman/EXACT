# src/reasoning/logic_semantic_parser.py

import re
from typing import List, Dict, Any
from src.reasoning.logic_ir import LogicProblem, LogicPremise, ReasoningTask, LogicHypothesis, MCQOption
from src.reasoning.parser import FOLParser

class LogicSemanticParser:
    """
    Parses natural language queries into structured LogicProblem IR tasks.
    Categorizes tasks as entailment, mcq, or minimal_premises.
    """
    def __init__(self):
        self.fol_parser = FOLParser()

    def parse(self, item: Dict[str, Any]) -> LogicProblem:
        problem = LogicProblem()
        
        # 1. Parse Premises
        raw_premises = item.get("premises-FOL", [])
        for i, p_str in enumerate(raw_premises):
            ast = self.fol_parser.parse_ast(p_str)
            problem.premises.append(LogicPremise(id=f"P{i+1}", text=p_str, ast=ast))
            
        # 2. Parse Task / Question
        # For this dataset, multiple questions might be present, we handle the first one for simplicity,
        # or we could return a list of LogicProblems. We'll extract the first.
        questions = item.get("questions", [])
        if not questions:
            return problem
            
        q_str = questions[0]
        q_lower = q_str.lower()
        
        task = ReasoningTask(task_type="unknown")
        
        if "fewest premises" in q_lower or "minimal" in q_lower:
            task.task_type = "minimal_premises"
        elif "which conclusion follows" in q_lower or "\na." in q_lower or "\nA." in q_str:
            task.task_type = "mcq"
        elif "does it follow" in q_lower or "is it true" in q_lower:
            task.task_type = "entailment"
            
        # Parse MCQ options if present
        if task.task_type in ["mcq", "minimal_premises"]:
            # Simple heuristic to extract A, B, C, D
            options = re.findall(r'([A-D])\.\s*(.+?)(?=[A-D]\.|$)', q_str, re.DOTALL)
            for opt_id, opt_text in options:
                opt_text = opt_text.strip()
                # For this dataset, options are NL. A true neuro-symbolic engine uses a lightweight
                # LLM translator here to convert NL -> FOL.
                # Since we bypass LLMs for deterministic steps, we assume the dataset provides or we heuristic it.
                # As a fallback, we store the text.
                task.options.append(MCQOption(option_id=opt_id, text=opt_text, ast=None))
                
        # Parse entailment target
        if task.task_type == "entailment":
            # Heuristic extraction of target (very difficult without LLM, requires dataset support)
            # We store the raw text
            task.target = LogicHypothesis(text=q_str, ast=None)
            
        problem.task = task
        return problem
