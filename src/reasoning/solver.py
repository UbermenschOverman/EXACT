# src/reasoning/solver.py

import logging
from typing import Dict, Any
from src.data.schema import ReasoningOutput
from src.reasoning.physics_semantic_parser import PhysicsSemanticParser
from src.reasoning.template_retriever import TemplateRetriever
from src.reasoning.physics_planner import PhysicsPlanner
from src.reasoning.logic_semantic_parser import LogicSemanticParser
from src.reasoning.mcq_solver import MCQSolver
from src.reasoning.logic_solver import LogicSolver

logger = logging.getLogger(__name__)

class EndToEndOrchestrator:
    """
    New neuro-symbolic orchestrator.
    Pipeline: Question -> Semantic Parser -> IR -> Retrieval -> Deterministic Solver -> Answer
    """
    def __init__(self):
        self.physics_parser = PhysicsSemanticParser()
        self.physics_retriever = TemplateRetriever()
        self.physics_planner = PhysicsPlanner()
        
        self.logic_parser = LogicSemanticParser()
        self.mcq_solver = MCQSolver()
        self.logic_solver = LogicSolver() # Handles standard Yes/No target queries

    def run(self, sample_data: Any) -> ReasoningOutput:
        """
        Generic entrypoint matching old Orchestrator signature.
        Routes to physics or logic solver based on input type/content.
        """
        if isinstance(sample_data, dict):
            return self.solve_logic(sample_data)
        elif isinstance(sample_data, str):
            # Assume physics string if it's raw text
            return self.solve_physics(sample_data)
        elif hasattr(sample_data, 'question'): # Sample object
            return self.solve_physics(sample_data.question)
        else:
            return ReasoningOutput(question=str(sample_data), answer="Unknown", valid=False)

    def solve_physics(self, question: str) -> ReasoningOutput:
        trace = []
        
        # 1. Semantic Parser -> IR Builder
        ir_problem = self.physics_parser.parse(question)
        trace.append({
            "action": "semantic_parse",
            "ir_entities": list(ir_problem.entities.keys()),
            "ir_relations": [r.relation_type for r in ir_problem.relations]
        })
        
        # 2. Structural Retrieval
        templates = self.physics_retriever.retrieve_physics_templates(ir_problem)
        trace.append({
            "action": "structural_retrieval",
            "templates": templates
        })
        
        # 3. Deterministic Solver (DAG)
        plan = self.physics_planner.plan_derivation(ir_problem)
        trace.append({
            "action": "dag_planning",
            "plan": plan
        })
        
        if not plan:
            return ReasoningOutput(question=question, answer="Unknown", error="No DAG plan found", valid=False)
            
        final_vars = self.physics_planner.execute_plan(ir_problem, plan)
        
        if not final_vars or ir_problem.targets[0].name not in final_vars:
            return ReasoningOutput(question=question, answer="Unknown", error="DAG execution failed", valid=False)
            
        target = ir_problem.targets[0]
        numeric_val = final_vars[target.name]
        unit = target.unit or ""
        answer_str = f"{numeric_val:.4g} {unit}".strip()
        
        trace.append({
            "action": "deterministic_solve",
            "result": answer_str
        })
        
        return ReasoningOutput(
            question=question,
            answer=answer_str,
            reasoning_trace=trace,
            confidence=1.0,
            valid=True
        )

    def solve_logic(self, item: Dict[str, Any]) -> ReasoningOutput:
        trace = []
        question_text = item.get("questions", [""])[0]
        
        # 1. Semantic Parser -> IR
        ir_problem = self.logic_parser.parse(item)
        trace.append({
            "action": "semantic_parse",
            "task_type": ir_problem.task.task_type
        })
        
        # 2. Deterministic Solver
        if ir_problem.task.task_type == "mcq":
            best_opt, scores = self.mcq_solver.evaluate(ir_problem)
            trace.append({
                "action": "mcq_evaluate",
                "scores": scores,
                "best_option": best_opt.option_id if best_opt else None
            })
            if best_opt:
                return ReasoningOutput(
                    question=question_text,
                    answer=best_opt.option_id,
                    reasoning_trace=trace,
                    confidence=1.0,
                    valid=True
                )
                
        elif ir_problem.task.task_type == "entailment":
            # Target fallback to string parsing
            target_text = ir_problem.task.target.text if ir_problem.task.target else ""
            # Assuming we extract the target FOL from text here or fallback
            return self.logic_solver.solve([p.text for p in ir_problem.premises], target_text)

        return ReasoningOutput(question=question_text, answer="Unknown", error="Unsupported task type", valid=False)