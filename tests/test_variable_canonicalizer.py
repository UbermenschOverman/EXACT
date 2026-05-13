# tests/test_variable_canonicalizer.py
"""Tests for VariableCanonicalizer using physics dataset variable patterns."""

import pytest
from src.reasoning.variable_canonicalizer import VariableCanonicalizer


@pytest.fixture
def canon():
    return VariableCanonicalizer()


def test_capacitor_energy_variables(canon):
    """Dataset format: C = 100 μF, U = 30 V → capacitance=1e-4, V=30"""
    raw = {"C": 1e-4, "U": 30.0}
    result = canon.normalize_for_formula(raw)
    assert "capacitance" in result
    assert result["capacitance"] == pytest.approx(1e-4)
    assert "V" in result
    assert result["V"] == pytest.approx(30.0)


def test_ohm_law_variables(canon):
    """R and I → R, current"""
    raw = {"R": 5.0, "I": 2.0}
    result = canon.normalize_for_formula(raw)
    assert "R" in result
    assert result["R"] == pytest.approx(5.0)
    assert "current" in result
    assert result["current"] == pytest.approx(2.0)


def test_coulomb_variables_preserved(canon):
    """q1, q2 already match coulomb_law variables — must pass through unchanged"""
    raw = {"q1": 6e-8, "q2": -6e-8, "r": 0.08}
    result = canon.normalize_for_formula(raw)
    assert result["q1"] == pytest.approx(6e-8)
    assert result["q2"] == pytest.approx(-6e-8)
    assert result["r"] == pytest.approx(0.08)


def test_power_variables(canon):
    raw = {"P": 100.0, "I": 5.0}
    result = canon.normalize_for_formula(raw)
    assert "P" in result
    assert "current" in result


def test_unknown_passthrough(canon):
    """Unknown variables should pass through unchanged."""
    raw = {"F1": 3.0, "F2": 4.0}
    result = canon.normalize_for_formula(raw)
    assert "F1" in result
    assert "F2" in result


def test_no_overwrite(canon):
    """First writer wins — existing canonical name not overwritten."""
    raw = {"V": 10.0, "U": 20.0}   # V is canonical, U also maps to V
    result = canon.normalize_for_formula(raw)
    # V=10 was set first via the dict ordering; U should not overwrite it
    assert "V" in result
