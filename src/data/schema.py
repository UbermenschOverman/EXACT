# src/data/schema.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class QASample:
    question: str
    answer: Optional[str] = None
    explanation: Optional[str] = None


@dataclass
class FOLSample(QASample):
    fol: Optional[str] = None


@dataclass
class PhysicsSample(QASample):
    reasoning: Optional[str] = None  # chain-of-thought
    formula: Optional[str] = None


@dataclass
class ReasoningOutput:
    """Unified output schema for all reasoning types."""
    question: str
    answer: str
    explanation: str = ""
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    used_premises: List[str] = field(default_factory=list)
    confidence: float = 0.0
    question_type: str = "unknown"   # "logic" | "physics" | "definition" | "unknown"
    valid: bool = False
    verification: Optional[Dict[str, Any]] = None
    attempt: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "explanation": self.explanation,
            "reasoning_trace": self.reasoning_trace,
            "used_premises": self.used_premises,
            "confidence": self.confidence,
            "question_type": self.question_type,
            "valid": self.valid,
            "verification": self.verification,
            "attempt": self.attempt,
            "error": self.error,
        }