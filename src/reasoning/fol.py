# src/reasoning/fol.py

from dataclasses import dataclass
from typing import List


@dataclass
class Predicate:
    name: str
    args: List[str]

    def __str__(self):
        return f"{self.name}({', '.join(self.args)})"


@dataclass
class Implication:
    premise: str
    conclusion: str

    def __str__(self):
        return f"{self.premise} -> {self.conclusion}"


@dataclass
class Quantifier:
    type: str  # "forall" or "exists"
    variable: str
    expression: str

    def __str__(self):
        return f"{self.type} {self.variable}: {self.expression}"