# tests/benchmark_physics_dataset.py

import csv
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.reasoning.physics_solver import PhysicsSolver

def run_benchmark(data_path: str):
    solver = PhysicsSolver()
    
    total = 0
    exact_matches = 0
    tol_matches = 0
    classification_correct = 0 # Not perfectly trackable without ground truth formula, but we track failures
    
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Format: id, question, cot, answer, unit
            # Only process 50 rows for this benchmark
            if total >= 50:
                break
                
            question = row['question']
            try:
                expected_ans = float(row['answer'])
            except ValueError:
                continue # Skip if no numeric answer
                
            result = solver.solve_question(question)
            
            total += 1
            
            if result.valid:
                # We consider it a classification success if it didn't fail
                classification_correct += 1
                
                # Check accuracy
                try:
                    # Strip unit
                    num_str = result.answer.split(' ')[0]
                    actual = float(num_str)
                    
                    if actual == expected_ans:
                        exact_matches += 1
                        tol_matches += 1
                    elif math.isclose(actual, expected_ans, rel_tol=1e-2):
                        tol_matches += 1
                        
                except Exception:
                    pass

    print("\n--- Physics Benchmark Results ---")
    print(f"Total Questions Evaluated: {total}")
    if total > 0:
        print(f"Formula Classification / Valid Extract: {classification_correct / total * 100:.2f}%")
        print(f"Tolerance Accuracy (rel_tol=1%): {tol_matches / total * 100:.2f}%")
        print(f"Exact Match Accuracy: {exact_matches / total * 100:.2f}%")

if __name__ == "__main__":
    data_path = os.path.join("data", "Physics_Problems_Text_Only.csv")
    if os.path.exists(data_path):
        run_benchmark(data_path)
    else:
        print(f"Dataset not found at {data_path}")
