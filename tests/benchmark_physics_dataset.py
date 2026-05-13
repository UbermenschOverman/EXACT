# tests/benchmark_physics_dataset.py
"""
Physics Benchmark — routes through EndToEndOrchestrator.
Granular failure categories + JSON diagnostics export.
"""

import csv, json, math, os, sys, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reasoning.solver import EndToEndOrchestrator


def run_benchmark(data_path: str, limit: int = 50):
    orchestrator = EndToEndOrchestrator()

    counters = {
        "total": 0,
        "extraction_fail": 0,
        "formula_fail": 0,
        "solve_fail": 0,
        "tol_match": 0,
        "exact_match": 0,
    }
    diagnostics = []

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

            result = orchestrator.run_physics(question)
            counters["total"] += 1

            diag = {
                "id": row.get("id", ""),
                "question": question[:80],
                "expected": expected,
                "predicted": result.answer,
                "valid": result.valid,
                "category": "unknown",
            }

            if not result.valid:
                trace_actions = [t.get("action", "") for t in result.reasoning_trace]
                has_vars = any(
                    t.get("action") == "canonicalize" and len(t.get("canonical", [])) > 0
                    for t in result.reasoning_trace
                )
                if has_vars:
                    counters["formula_fail"] += 1
                    diag["category"] = "formula_fail"
                else:
                    counters["extraction_fail"] += 1
                    diag["category"] = "extraction_fail"
            else:
                try:
                    actual = float(result.answer.split()[0])
                except (ValueError, IndexError):
                    counters["solve_fail"] += 1
                    diag["category"] = "solve_fail"
                    diagnostics.append(diag)
                    continue

                diag["actual_numeric"] = actual
                if actual == expected:
                    counters["exact_match"] += 1
                    counters["tol_match"] += 1
                    diag["category"] = "exact_match"
                elif math.isclose(actual, expected, rel_tol=1e-2):
                    counters["tol_match"] += 1
                    diag["category"] = "tol_match"
                else:
                    diag["category"] = "wrong_answer"

            diagnostics.append(diag)

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
        wrong = n - counters['extraction_fail'] - counters['formula_fail'] - counters['solve_fail'] - counters['tol_match']
        print(f"  Wrong answer (numeric)    : {max(0,wrong)} ({max(0,wrong)/n*100:.1f}%)")
    print("══════════════════════════════════════════")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/physics_diagnostics.json", "w") as f:
        json.dump({"counters": counters, "details": diagnostics}, f, indent=2, ensure_ascii=False)
    print(f"  Diagnostics → outputs/physics_diagnostics.json")


if __name__ == "__main__":
    path = os.path.join("data", "Physics_Problems_Text_Only.csv")
    if os.path.exists(path):
        run_benchmark(path)
    else:
        print(f"Dataset not found: {path}")
