# tests/test_fol_ast_parser.py

import pytest
import os
import json
from src.reasoning.parser import FOLParser
from src.reasoning.fol import ForAll, Exists, Implication, Predicate, Variable, Constant

def test_fol_parser_ast_nodes():
    parser = FOLParser()
    
    # ForAll and Implication
    ast = parser.parse_ast("∀x (WT(x) → O(x))")
    assert isinstance(ast, ForAll)
    assert ast.variable.name == "x"
    assert isinstance(ast.statement, Implication)
    assert isinstance(ast.statement.premise, Predicate)
    assert ast.statement.premise.name == "WT"
    assert isinstance(ast.statement.premise.args[0], Variable)
    
    # Exists
    ast = parser.parse_ast("∃x (BP(x))")
    assert isinstance(ast, Exists)
    assert ast.variable.name == "x"
    assert isinstance(ast.statement, Predicate)
    
    # Structural normalization
    ast = parser.parse_ast("forall(x, O(x) -> CR(x))")
    assert isinstance(ast, ForAll)
    assert isinstance(ast.statement, Implication)

def test_dataset_parser_benchmark():
    """
    Measures the percentage of successfully parsed premises from the Logic dataset.
    """
    parser = FOLParser()
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "Logic_Based_Educational_Queries.json")
    
    if not os.path.exists(data_path):
        pytest.skip(f"Dataset not found at {data_path}")
        
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total_premises = 0
    successfully_parsed = 0
    
    for item in data:
        for p_str in item.get("premises-FOL", []):
            total_premises += 1
            try:
                ast = parser.parse_ast(p_str)
                if ast is not None:
                    successfully_parsed += 1
            except Exception:
                pass
                
    accuracy = (successfully_parsed / total_premises) * 100 if total_premises > 0 else 0
    print(f"\nDataset Parser Benchmark: {accuracy:.2f}% ({successfully_parsed}/{total_premises})")
    
    # We want a high accuracy for competition grade
    assert accuracy > 80.0
