# tests/test_logic_dataset_benchmark.py

import json
import os
import sys

# Add parent directory to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reasoning.logic_solver import LogicSolver

def run_benchmark(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    solver = LogicSolver()
    
    total = 0
    correct_yesno = 0
    total_yesno = 0
    correct_mcq = 0
    total_mcq = 0
    
    total_depth = 0
    
    for item in data[:50]:  # Limit to 50 for quick benchmark, remove for full
        premises = item.get("premises-FOL", [])
        questions = item.get("questions", [])
        answers = item.get("answers", [])
        
        for q_idx, q_str in enumerate(questions):
            expected_ans = answers[q_idx]
            
            # Simple heuristic to determine if MCQ or Yes/No
            is_mcq = "A." in q_str and "B." in q_str
            
            if is_mcq:
                total_mcq += 1
                # MCQ logic: Evaluate each option as a target. This is complex
                # For this benchmark, we just mark it Unknown if we don't have full MCQ support yet
                total += 1
                continue
            else:
                total_yesno += 1
                
                # Extract target from Yes/No question
                # "Does it follow that if all Python projects are well-structured, then all Python projects are optimized, according to the premises?"
                # This requires LLM to convert question to FOL. For benchmark, we just assume we have the target FOL.
                # Since the dataset doesn't provide target FOL for the questions, we have a gap here!
                # For the sake of the benchmark skeleton, we will pass a dummy string and test the pipeline
                target_fol = "∀x (WS(x) → O(x))" # Dummy for now
                
                result = solver.solve(premises, target_fol)
                
                if result.answer.lower() == expected_ans.lower():
                    correct_yesno += 1
                
                total_depth += len(result.reasoning_trace)
            
            total += 1

    print("\n--- Logic Benchmark Results ---")
    print(f"Total Questions Evaluated: {total}")
    if total_yesno > 0:
        print(f"Yes/No Accuracy: {correct_yesno / total_yesno * 100:.2f}%")
    if total_mcq > 0:
        print(f"MCQ Accuracy: {correct_mcq / total_mcq * 100:.2f}%")
        
    avg_depth = total_depth / total if total > 0 else 0
    print(f"Average Inference Depth: {avg_depth:.2f} steps")

if __name__ == "__main__":
    data_path = os.path.join("data", "Logic_Based_Educational_Queries.json")
    if os.path.exists(data_path):
        run_benchmark(data_path)
    else:
        print(f"Dataset not found at {data_path}")
