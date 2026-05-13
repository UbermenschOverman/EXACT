# src/reasoning/llm_contracts.py
"""
Strict JSON schemas for LLM Compiler outputs.
The LLM is ONLY allowed to produce these schemas.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class PhysicsCompilationSchema:
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    goals: List[Dict[str, Any]] = field(default_factory=list)
    task_type: str = "physics"

    @classmethod
    def from_json(cls, raw: str) -> Optional["PhysicsCompilationSchema"]:
        try:
            d = json.loads(raw) if isinstance(raw, str) else raw
            return cls(
                entities=d.get("entities", []),
                relations=d.get("relations", []),
                goals=d.get("goals", []),
                task_type=d.get("task_type", "physics"),
            )
        except Exception:
            return None

    def is_valid(self) -> bool:
        return bool(self.entities or self.relations)


@dataclass
class LogicTargetSchema:
    task_type: str = "unknown"
    fol_target: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0

    @classmethod
    def from_json(cls, raw: str) -> Optional["LogicTargetSchema"]:
        try:
            d = json.loads(raw) if isinstance(raw, str) else raw
            return cls(
                task_type=d.get("task_type", "unknown"),
                fol_target=d.get("fol_target"),
                entities=d.get("entities", []),
                confidence=float(d.get("confidence", 0.5)),
            )
        except Exception:
            return None

    def is_valid(self) -> bool:
        return self.task_type in ("entailment", "mcq", "unknown")
