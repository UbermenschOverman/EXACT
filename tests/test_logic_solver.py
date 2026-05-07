# tests/test_logic_solver.py

import pytest
from src.reasoning.logic_solver import LogicSolver
from src.reasoning.parser import FOLParser
from src.reasoning.premise_graph import PremiseGraph

def test_fol_parser_implication():
    parser = FOLParser()
    parsed = parser.parse("If it rains, then the ground is wet")
    assert parsed["type"] == "implication"
    assert parsed["premise"] == "it rains"
    assert parsed["conclusion"] == "the ground is wet"

def test_fol_parser_extract():
    parser = FOLParser()
    text = "1. If all birds can fly, then Tweety can fly.\n2. All birds can fly."
    extracted = parser.extract_premises_and_rules(text)
    
    assert len(extracted["premises"]) == 2
    assert len(extracted["rules"]) == 1
    
    assert extracted["rules"][0]["conditions"][0] == "P1"
    assert extracted["rules"][0]["conclusion_text"] == "Tweety can fly."

def test_logic_solver_forward_chaining():
    solver = LogicSolver()
    premises = [
        {"id": "P1", "text": "All birds can fly"},
    ]
    rules = [
        {"id": "R1", "conditions": ["P1"], "conclusion": "P2", "text": "If all birds can fly, then Tweety can fly"}
    ]
    result = solver.solve(premises, rules, target="P2")
    
    assert result["answer"] != "UNKNOWN"
    assert "P2" in result["derived"]
    assert len(result["reasoning_trace"]) > 0

def test_logic_solver_contradiction():
    solver = LogicSolver()
    premises = [
        {"id": "P1", "text": "Tweety can fly"},
        {"id": "P2", "text": "not Tweety can fly"},
    ]
    rules = []
    result = solver.solve(premises, rules)
    
    assert result["contradiction"] is not None
    assert "Contradiction" in result["contradiction"]
    assert result["confidence"] <= 0.1
