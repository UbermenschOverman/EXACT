# src/reasoning/fol.py

from dataclasses import dataclass
from typing import List, Union

class ASTNode:
    def __str__(self):
        return ""

@dataclass
class Variable(ASTNode):
    name: str
    def __str__(self):
        return self.name
    def __hash__(self):
        return hash(("Var", self.name))

@dataclass
class Constant(ASTNode):
    name: str
    def __str__(self):
        return self.name
    def __hash__(self):
        return hash(("Const", self.name))

@dataclass
class Predicate(ASTNode):
    name: str
    args: List[Union[Variable, Constant]]
    def __str__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"
    def __hash__(self):
        return hash((self.name, tuple(self.args)))

@dataclass
class Negation(ASTNode):
    statement: ASTNode
    def __str__(self):
        return f"¬{self.statement}"

@dataclass
class Conjunction(ASTNode):
    left: ASTNode
    right: ASTNode
    def __str__(self):
        return f"({self.left} ∧ {self.right})"

@dataclass
class Implication(ASTNode):
    premise: ASTNode
    conclusion: ASTNode
    def __str__(self):
        return f"({self.premise} → {self.conclusion})"

@dataclass
class ForAll(ASTNode):
    variable: Variable
    statement: ASTNode
    def __str__(self):
        return f"∀{self.variable} ({self.statement})"

@dataclass
class Exists(ASTNode):
    variable: Variable
    statement: ASTNode
    def __str__(self):
        return f"∃{self.variable} ({self.statement})"