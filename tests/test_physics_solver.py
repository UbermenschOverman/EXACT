# tests/test_physics_solver.py

import pytest
from src.reasoning.physics_solver import PhysicsSolver
from src.reasoning.formula_bank import rank_formulas
from src.reasoning.variable_extractor import VariableExtractor


def test_variable_extraction():
    extractor = VariableExtractor()

    # Variable extractor requires explicit "Var = Value Unit" format
    text = "R = 5 Ω and V = 10 V. Find the current."

    vars_extracted = extractor.extract(text)

    assert "R" in vars_extracted
    assert vars_extracted["R"] == pytest.approx(5.0)
    assert "V" in vars_extracted
    assert vars_extracted["V"] == pytest.approx(10.0)


def test_detect_formula():
    text = "Using Ohm law, find the current through a resistor with voltage and resistance given."

    # rank_formulas scores by keyword match; ohm_law should rank first
    ranked = rank_formulas(text)
    assert len(ranked) > 0
    assert ranked[0][0] == "ohm_law"


def test_physics_solver_ohm_law():
    solver = PhysicsSolver()

    # Use explicit variable format so VariableExtractor can parse it
    result = solver.solve_question(
        "R = 5 Ω and V = 10 V. What current flows through the resistor?"
    )

    assert result.valid
    assert result.confidence > 0.5


def test_physics_solver_power():
    solver = PhysicsSolver()

    result = solver.solve_question(
        "P = 100 W and current = 5 A. Determine the voltage."
    )

    assert result.valid
    assert result.confidence > 0.5


def test_physics_solver_capacitor():
    solver = PhysicsSolver()

    result = solver.solve_question(
        "capacitance = 0.01 F and V = 100 V. Compute the stored energy."
    )

    assert result.valid
    assert result.confidence > 0.5