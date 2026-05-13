# tests/test_semantic_compiler.py
"""Tests for SemanticCompiler using real dataset formats."""

import json
import os
import pytest

from src.reasoning.semantic_compiler import SemanticCompiler


@pytest.fixture
def compiler():
    return SemanticCompiler()


# ── Physics ──────────────────────────────────────────────────────────────────

def test_physics_explicit_format(compiler):
    """Dataset format: 'C = 100 μF and U = 30 V'"""
    text = "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
    wm = compiler.compile_physics(text)
    flat = wm.flat_quantities()

    # Should extract C and U
    assert len(flat) >= 2, f"Expected ≥2 vars, got: {flat}"
    assert "C" in flat or "capacitance" in flat
    assert "U" in flat or "V" in flat


def test_physics_target_detection(compiler):
    """Target 'energy' should be detected from 'energy stored'"""
    text = "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
    wm = compiler.compile_physics(text)
    assert wm.goals, "Should have at least one goal"
    assert any(g.target in ("energy", "E") for g in wm.goals)


def test_physics_ohm_law(compiler):
    text = "R = 5 Ω and V = 10 V. Find the current."
    wm = compiler.compile_physics(text)
    flat = wm.flat_quantities()
    assert "R" in flat or "resistance" in flat
    assert "V" in flat or "voltage" in flat


# ── Logic ────────────────────────────────────────────────────────────────────

def test_logic_mcq_item(compiler):
    """Dataset item 0: MCQ with 4 options"""
    item = {
        "premises-FOL": ["∀x (WT(x) → O(x))", "∀x (WT(x))"],
        "questions": [
            "Which conclusion follows with the fewest premises?\n"
            "A. If a Python project is not optimized, then it is not well-tested\n"
            "B. If all Python projects are optimized, then all Python projects are well-structured\n"
            "C. If a Python project is well-tested, then it must be clean and readable\n"
            "D. If a Python project is not optimized, then it does not follow PEP 8 standards"
        ],
    }
    wm = compiler.compile_logic(item)
    assert wm.problem_type == "logic"
    mcq_goal = next((g for g in wm.goals if g.goal_type == "mcq"), None)
    assert mcq_goal is not None, "Should detect MCQ goal"
    assert len(mcq_goal.options) == 4


def test_logic_yesno_scholarship(compiler):
    """Dataset item 1 Yes/No: 'Does Sophia qualify for the university scholarship?'"""
    item = {
        "premises-FOL": [
            "ForAll(x, (completed_core_curriculum(x) ∧ passed_science_assessment(x)) → qualified_for_advanced_courses(x))",
        ],
        "questions": [
            "Does Sophia qualify for the university scholarship, according to the premises?"
        ],
    }
    wm = compiler.compile_logic(item)
    entailment = next((g for g in wm.goals if g.goal_type == "entailment"), None)
    assert entailment is not None, "Should find entailment goal for scholarship question"
    assert "qualifies_for_scholarship" in (entailment.target or "")


def test_logic_mcq_options_canonicalized(compiler):
    """MCQ option 'Sophia qualifies for the university scholarship' should canonicalize."""
    item = {
        "premises-FOL": ["∀x (WT(x) → O(x))"],
        "questions": [
            "Which is the strongest conclusion?\n"
            "A. Sophia qualifies for the university scholarship\n"
            "B. Sophia needs a faculty recommendation\n"
            "C. Sophia is eligible for the international program\n"
            "D. Sophia needs to pass the language proficiency exam"
        ],
    }
    wm = compiler.compile_logic(item)
    mcq = next((g for g in wm.goals if g.goal_type == "mcq"), None)
    assert mcq is not None
    # At least one option should have a FOL string
    canon_count = sum(1 for o in mcq.options if o.fol_str)
    assert canon_count >= 1, f"Expected ≥1 canonicalized option, got {canon_count}"
