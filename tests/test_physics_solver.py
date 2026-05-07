# tests/test_physics_solver.py

import pytest
from src.reasoning.physics_solver import PhysicsSolver
from src.reasoning.formula_bank import detect_formula
from src.reasoning.variable_extractor import VariableExtractor

def test_variable_extraction():
    extractor = VariableExtractor()
    text = "Calculate current if V=10V and R=5 ohm."
    vars_extracted = extractor.extract(text)
    assert vars_extracted == {"V": 10.0, "R": 5.0}

    target = extractor.extract_target(text)
    assert target == "I"

def test_detect_formula():
    text = "Calculate current if V=10V and R=5 ohm."
    formula_id = detect_formula(text)
    assert formula_id == "ohm_law"

def test_physics_solver_ohm_law():
    solver = PhysicsSolver()
    result = solver.solve_question("Calculate current if V=10V and R=5 ohm.")
    assert result["confidence"] > 0.5
    assert result["numeric_value"] == 2.0
    assert result["unit"] == "A"
    assert result["formula_id"] == "ohm_law"

def test_physics_solver_power():
    solver = PhysicsSolver()
    result = solver.solve_question("If a circuit has power P=100W and current I=5A, what is the voltage?")
    assert result["confidence"] > 0.5
    assert result["numeric_value"] == 20.0
    assert result["unit"] == "V"
    assert result["formula_id"] == "power_vi"

def test_physics_solver_capacitor():
    solver = PhysicsSolver()
    result = solver.solve_question("A capacitor with C=0.01F is charged to V=100V. What is the stored energy?")
    assert result["confidence"] > 0.5
    assert result["numeric_value"] == 50.0
    assert result["unit"] == "J"
    assert result["formula_id"] == "capacitor_energy"
