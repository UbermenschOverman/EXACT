# tests/benchmark_physics_dataset.py
"""
Physics Benchmark — routes through EndToEndOrchestrator.
Reports granular failures: extraction, compilation, formula, solve.
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reasoning.solver import EndToEndOrchestrator


def run_benchmark(data_path: str, limit: int = 50):
    orchestrator = EndToEndOrchestrator()

    counters = {
        "total": 0,
        "extraction_fail": 0,    # WorldModel has no quantities
        "formula_fail": 0,       # no formula matched
        "solve_fail": 0,         # formula matched but SymPy failed
        "tol_match": 0,          # within 1% tolerance
        "exact_match": 0,        # exact numeric match
    }

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if counters["total"] >= limit:
                break

            question = row["question"]
            try:
                expected = float(row["answer"])
            except (ValueError, KeyError):
                continue

            # Run full pipeline
            result = orchestrator.run_physics(question)
            counters["total"] += 1

            if not result.valid:
                # Classify failure
                trace_actions = [t.get("action", "") for t in result.reasoning_trace]
                if "world_model_vars" in trace_actions:
                    # compilation ran but no formula found
                    has_vars = any(
                        "Flattened variables: {}" not in t.get("detail", "")
                        for t in result.reasoning_trace
                        if t.get("action") == "world_model_vars"
                    )
                    if has_vars:
                        counters["formula_fail"] += 1
                    else:
                        counters["extraction_fail"] += 1
                else:
                    counters["extraction_fail"] += 1
                continue

            # Parse numeric answer
            try:
                actual = float(result.answer.split()[0])
            except (ValueError, IndexError):
                counters["solve_fail"] += 1
                continue

            if actual == expected:
                counters["exact_match"] += 1
                counters["tol_match"] += 1
            elif math.isclose(actual, expected, rel_tol=1e-2):
                counters["tol_match"] += 1

    n = counters["total"]
    print("\n══════════════════════════════════════════")
    print("   EXACT Physics Benchmark Results")
    print("══════════════════════════════════════════")
    print(f"  Total evaluated           : {n}")
    if n > 0:
        print(f"  Extraction failures       : {counters['extraction_fail']} ({counters['extraction_fail']/n*100:.1f}%)")
        print(f"  Formula match failures    : {counters['formula_fail']} ({counters['formula_fail']/n*100:.1f}%)")
        print(f"  Solve failures            : {counters['solve_fail']} ({counters['solve_fail']/n*100:.1f}%)")
        print(f"  Tolerance accuracy (1%)   : {counters['tol_match']} ({counters['tol_match']/n*100:.1f}%)")
        print(f"  Exact match accuracy      : {counters['exact_match']} ({counters['exact_match']/n*100:.1f}%)")
    print("══════════════════════════════════════════")


if __name__ == "__main__":
    path = os.path.join("data", "Physics_Problems_Text_Only.csv")
    if os.path.exists(path):
        run_benchmark(path)
    else:
        print(f"Dataset not found: {path}")
