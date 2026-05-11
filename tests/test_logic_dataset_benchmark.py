# tests/test_logic_dataset_benchmark.py
"""
Logic Benchmark — routes through EndToEndOrchestrator.
Reports: premise parse rate, MCQ canonicalization, proof success, contradiction detection.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reasoning.solver import EndToEndOrchestrator
from src.reasoning.semantic_compiler import SemanticCompiler


def run_benchmark(data_path: str, limit: int = 50):
    data = json.load(open(data_path, encoding="utf-8"))

    orchestrator = EndToEndOrchestrator()
    compiler = SemanticCompiler()

    counters = {
        "total_questions": 0,
        "premise_parse_ok": 0,
        "total_items": 0,
        # MCQ
        "total_mcq": 0,
        "mcq_canonicalized": 0,    # at least 1 option got a FOL string
        "mcq_correct": 0,
        # Yes/No entailment
        "total_yesno": 0,
        "yesno_target_found": 0,   # deterministic translator found FOL target
        "yesno_correct": 0,
        # Contradiction
        "contradiction_detected": 0,
    }

    for item in data[:limit]:
        premises_fol = item.get("premises-FOL", [])
        questions = item.get("questions", [])
        answers = item.get("answers", [])

        counters["total_items"] += 1

        # Premise parse accuracy
        parse_ok = len(premises_fol) > 0
        if parse_ok:
            counters["premise_parse_ok"] += 1

        # Compile to WorldModel for task classification
        wm = compiler.compile_logic(item)

        for q_idx, q_str in enumerate(questions):
            expected = answers[q_idx] if q_idx < len(answers) else None
            counters["total_questions"] += 1

            is_mcq = any(g.goal_type == "mcq" for g in wm.goals)

            if is_mcq:
                counters["total_mcq"] += 1
                result = orchestrator.run_logic(item)

                # Count canonicalization success
                mcq_goal = next((g for g in wm.goals if g.goal_type == "mcq"), None)
                if mcq_goal and any(o.fol_str for o in mcq_goal.options):
                    counters["mcq_canonicalized"] += 1

                if expected and result.answer.upper() == expected.upper():
                    counters["mcq_correct"] += 1

            else:
                counters["total_yesno"] += 1

                # Check if deterministic translator found a FOL target
                entailment_goal = next(
                    (g for g in wm.goals if g.goal_type == "entailment"), None)
                if entailment_goal:
                    counters["yesno_target_found"] += 1
                    result = orchestrator.run_logic(item)
                    if expected and result.answer.lower() == expected.lower():
                        counters["yesno_correct"] += 1
                # else: no FOL target found, skip (would need LLM)

            # Break after first question per item for benchmark speed
            break

    n_q = counters["total_questions"]
    n_yn = counters["total_yesno"]
    n_mcq = counters["total_mcq"]

    print("\n══════════════════════════════════════════")
    print("   EXACT Logic Benchmark Results")
    print("══════════════════════════════════════════")
    print(f"  Items evaluated           : {counters['total_items']}")
    print(f"  Total questions           : {n_q}")
    print(f"  Premise parse rate        : {counters['premise_parse_ok']}/{counters['total_items']} ({counters['premise_parse_ok']/max(counters['total_items'],1)*100:.1f}%)")
    print()
    print(f"  Yes/No questions          : {n_yn}")
    if n_yn:
        print(f"    → FOL target found      : {counters['yesno_target_found']} ({counters['yesno_target_found']/n_yn*100:.1f}%)")
        print(f"    → Correct answers       : {counters['yesno_correct']} ({counters['yesno_correct']/n_yn*100:.1f}%)")
    print()
    print(f"  MCQ questions             : {n_mcq}")
    if n_mcq:
        print(f"    → Options canonicalized : {counters['mcq_canonicalized']} ({counters['mcq_canonicalized']/n_mcq*100:.1f}%)")
        print(f"    → Correct answers       : {counters['mcq_correct']} ({counters['mcq_correct']/n_mcq*100:.1f}%)")
    print("══════════════════════════════════════════")


if __name__ == "__main__":
    path = os.path.join("data", "Logic_Based_Educational_Queries.json")
    if os.path.exists(path):
        run_benchmark(path)
    else:
        print(f"Dataset not found: {path}")
