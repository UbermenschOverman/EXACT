# tests/test_logic_solver.py

import pytest
from src.reasoning.logic_solver import LogicSolver
from src.reasoning.parser import FOLParser
from src.reasoning.premise_graph import PremiseGraph


def test_fol_parser_implication():
    parser = FOLParser()

    parsed = parser.parse(
        "Whenever it rains heavily, the streets become wet."
    )

    assert parsed["type"] == "implication"

    # robust semantic-style assertions
    assert "rains" in parsed["premise"].lower()
    assert "wet" in parsed["conclusion"].lower()


def test_fol_parser_extract():
    parser = FOLParser()

    text = (
        "1. Every bird is capable of flight.\n"
        "2. Tweety is a bird.\n"
        "3. If Tweety is a bird, then Tweety can fly."
    )

    extracted = parser.extract_premises_and_rules(text)

    assert len(extracted["premises"]) >= 2
    assert len(extracted["rules"]) >= 1

    rule = extracted["rules"][0]

    assert len(rule["conditions"]) > 0
    assert "Tweety" in rule["conclusion_text"]


def test_logic_solver_forward_chaining():
    solver = LogicSolver()

    premises = [
        {"id": "P1", "text": "All mammals are warm blooded"},
        {"id": "P2", "text": "A dolphin is a mammal"},
    ]

    rules = [
        {
            "id": "R1",
            "conditions": ["P1", "P2"],
            "conclusion": "P3",
            "text": (
                "If all mammals are warm blooded and "
                "a dolphin is a mammal, then a dolphin is warm blooded"
            ),
        }
    ]

    result = solver.solve(premises, rules, target="P3")

    assert result["answer"] != "UNKNOWN"

    assert "P3" in result["derived"]

    assert len(result["reasoning_trace"]) > 0


def test_logic_solver_multi_hop_reasoning():
    solver = LogicSolver()

    premises = [
        {"id": "P1", "text": "All students who study pass exams"},
        {"id": "P2", "text": "Alice studies regularly"},
    ]

    rules = [
        {
            "id": "R1",
            "conditions": ["P1", "P2"],
            "conclusion": "P3",
            "text": (
                "If students study regularly, they pass exams"
            ),
        },
        {
            "id": "R2",
            "conditions": ["P3"],
            "conclusion": "P4",
            "text": (
                "If Alice passes exams, then Alice graduates"
            ),
        },
    ]

    result = solver.solve(premises, rules, target="P4")

    assert result["answer"] != "UNKNOWN"

    assert "P4" in result["derived"]

    # important robustness signal:
    # did the chain actually propagate?
    assert len(result["reasoning_trace"]) >= 2


def test_logic_solver_contradiction():
    solver = LogicSolver()

    premises = [
        {"id": "P1", "text": "Penguins can fly"},
        {"id": "P2", "text": "not Penguins can fly"},
    ]

    rules = []

    result = solver.solve(premises, rules)

    assert result["contradiction"] is not None

    assert "Contradiction" in result["contradiction"]

    assert result["confidence"] <= 0.1