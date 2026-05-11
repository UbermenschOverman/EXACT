# src/reasoning/logic_ir.py

from dataclasses import dataclass, field
from typing import List, Optional, Any
from src.reasoning.fol import ASTNode

@dataclass
class LogicPremise:
    id: str
    text: str
    ast: Optional[ASTNode] = None

@dataclass
class LogicHypothesis:
    text: str
    ast: Optional[ASTNode] = None

@dataclass
class MCQOption:
    option_id: str # e.g., "A", "B"
    text: str
    ast: Optional[ASTNode] = None

@dataclass
class ReasoningTask:
    task_type: str # "entailment", "mcq", "minimal_premises", "contradiction"
    target: Optional[LogicHypothesis] = None
    options: List[MCQOption] = field(default_factory=list)

@dataclass
class LogicProblem:
    premises: List[LogicPremise] = field(default_factory=list)
    task: Optional[ReasoningTask] = None
