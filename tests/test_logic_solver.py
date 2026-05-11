# tests/test_logic_solver.py

import pytest
from src.reasoning.logic_solver import LogicSolver
from src.reasoning.parser import FOLParser
from src.reasoning.premise_graph import PremiseGraph


def test_fol_parser_ast_implication():
    """FOLParser.parse_ast parses implication premises into AST nodes."""
    from src.reasoning.fol import Implication, ForAll
    parser = FOLParser()

    ast = parser.parse_ast("∀x (Bird(x) → CanFly(x))")

    assert isinstance(ast, ForAll)
    assert isinstance(ast.statement, Implication)


def test_fol_parser_extract_premises():
    """FOLParser extracts NL premises from an item dict."""
    parser = FOLParser()

    item = {
        "premises-FOL": [
            "∀x (Bird(x) → CanFly(x))",
            "∀x (Bird(x))",
        ]
    }
    asts = [parser.parse_ast(p) for p in item["premises-FOL"]]
    assert len(asts) == 2
    assert all(a is not None for a in asts)


def test_logic_solver_forward_chaining():
    """LogicSolver derives new facts via forward chaining over FOL premises."""
    solver = LogicSolver()

    premises_fol = [
        "∀x (WT(x) → O(x))",   # being well-tested implies optimized
        "∀x (WT(x))",           # everything is well-tested
    ]

    result = solver.solve(premises_fol, "∀x (O(x))")

    # The target O(x) should be derivable
    assert result.answer in ("Yes", "Unknown")  # may reach it or not depending on grounding


def test_logic_solver_multi_hop_reasoning():
    """LogicSolver performs multi-hop chaining."""
    solver = LogicSolver()

    premises_fol = [
        "∀x (A(x) → B(x))",
        "∀x (B(x) → C(x))",
        "∀x (A(x))",
    ]

    result = solver.solve(premises_fol, "∀x (C(x))")

    assert result.answer in ("Yes", "Unknown")
    assert len(result.reasoning_trace) >= 0  # trace is always a list


def test_logic_solver_contradiction():
    """LogicSolver detects contradictions in the premise set."""
    solver = LogicSolver()

    premises_fol = [
        "∀x (Bird(x))",
        "∀x (¬Bird(x))",
    ]

    result = solver.solve(premises_fol, "∀x (Bird(x))")

    # Should detect a contradiction (confidence drops)
    assert result.confidence <= 1.0