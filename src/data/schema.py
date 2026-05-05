# src/data/schema.py

from dataclasses import dataclass
from typing import Optional, List


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
class ModelOutput:
    question: str
    answer: str
    explanation: Optional[str]
    fol: Optional[str]
    proof: Optional[List[str]]
    valid: bool