# tests/test_variable_extractor_dataset.py

import pytest
from src.reasoning.variable_extractor import VariableExtractor

def test_variable_extractor_scientific_notation():
    extractor = VariableExtractor()
    
    # 10^-8
    res = extractor.extract("q = 10^-8 C")
    assert pytest.approx(res["charge"]) == 1e-8
    
    # 10⁻⁸
    res = extractor.extract("q = 10⁻⁸ C")
    assert pytest.approx(res["charge"]) == 1e-8
    
    # 3 × 10^-6
    res = extractor.extract("q = 3 × 10^-6 C")
    assert pytest.approx(res["charge"]) == 3e-6
    
    # 3x10^-6
    res = extractor.extract("q = 3x10^-6 C")
    assert pytest.approx(res["charge"]) == 3e-6
    
    # 3e-6
    res = extractor.extract("q = 3e-6 C")
    assert pytest.approx(res["charge"]) == 3e-6

def test_variable_extractor_si_prefixes():
    extractor = VariableExtractor()
    
    # μF
    res = extractor.extract("C = 100 μF")
    assert pytest.approx(res["capacitance"]) == 100e-6
    
    # mC
    res = extractor.extract("Q = 5 mC")
    assert pytest.approx(res["charge"]) == 5e-3
    
    # kΩ
    res = extractor.extract("R = 2 kΩ")
    assert pytest.approx(res["R"]) == 2000
    
    # nC
    res = extractor.extract("q = 8 nC")
    assert pytest.approx(res["charge"]) == 8e-9

def test_variable_extractor_canonical_aliases():
    extractor = VariableExtractor()
    
    # U -> V
    res = extractor.extract("U = 30 V")
    assert "V" in res
    assert res["V"] == 30.0
    
    # I -> current
    res = extractor.extract("I = 5 A")
    assert "current" in res
    assert res["current"] == 5.0
