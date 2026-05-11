# tests/test_physics_solver.py

import pytest
from src.reasoning.physics_solver import PhysicsSolver
from src.reasoning.formula_bank import detect_formula
from src.reasoning.variable_extractor import VariableExtractor


def test_variable_extraction():
    extractor = VariableExtractor()

    text = (
        "A 5 ohm resistor is connected across a 10 volt battery. "
        "Determine the current flowing through the circuit."
    )

    vars_extracted = extractor.extract(text)

    assert vars_extracted == {"R": 5.0, "V": 10.0}

    target = extractor.extract_target(text)
    assert target == "I"


def test_detect_formula():
    text = (
        "A resistor of 5Ω has 10V across it. "
        "Using Ohm's law, find the current."
    )

    formula_id = detect_formula(text)

    assert formula_id == "ohm_law"


def test_physics_solver_ohm_law():
    solver = PhysicsSolver()

    result = solver.solve_question(
        "A resistor of 5 ohms is connected to a 10V source. "
        "What current flows through the resistor?"
    )

    assert result["confidence"] > 0.5
    assert result["numeric_value"] == 2.0
    assert result["unit"] == "A"
    assert result["formula_id"] == "ohm_law"


def test_physics_solver_power():
    solver = PhysicsSolver()

    result = solver.solve_question(
        "An electrical device consumes 100 watts of power while "
        "carrying 5 amperes of current. Determine the voltage."
    )

    assert result["confidence"] > 0.5
    assert result["numeric_value"] == 20.0
    assert result["unit"] == "V"
    assert result["formula_id"] == "power_vi"


def test_physics_solver_capacitor():
    solver = PhysicsSolver()

    result = solver.solve_question(
        "A capacitor with capacitance 0.01 farads is charged "
        "to 100 volts. Compute the stored energy."
    )

    assert result["confidence"] > 0.5
    assert result["numeric_value"] == 50.0
    assert result["unit"] == "J"
    assert result["formula_id"] == "capacitor_energy"