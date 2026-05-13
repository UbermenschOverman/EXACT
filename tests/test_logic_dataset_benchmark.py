# tests/test_logic_dataset_benchmark.py
"""
Logic Benchmark — routes ALL questions through EndToEndOrchestrator.
Reports granular: premise parse, MCQ canonicalization, FOL target, proof success.
Exports JSON diagnostics.
"""

import json, os, sys, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reasoning.solver import EndToEndOrchestrator
from src.reasoning.semantic_compiler import SemanticCompiler


def run_benchmark(data_path: str, limit: int = 50):
    data = json.load(open(data_path, encoding="utf-8"))
    orchestrator = EndToEndOrchestrator()
    compiler = SemanticCompiler()

    counters = {
        "total_items": 0, "total_questions": 0,
        "premise_parse_ok": 0,
        "total_mcq": 0, "mcq_canonicalized": 0, "mcq_correct": 0,
        "total_yesno": 0, "yesno_target_found": 0, "yesno_correct": 0,
    }
    diagnostics = []

    for item in data[:limit]:
        premises_fol = item.get("premises-FOL", [])
        questions = item.get("questions", [])
        answers = item.get("answers", [])
        counters["total_items"] += 1

        if len(premises_fol) > 0:
            counters["premise_parse_ok"] += 1

        # Evaluate EVERY question in this item
        for q_idx, q_text in enumerate(questions):
            expected = answers[q_idx] if q_idx < len(answers) else None
            counters["total_questions"] += 1

            # Build per-question item
            q_item = dict(item)
            q_item["questions"] = [q_text]

            wm = compiler.compile_logic(q_item)
            is_mcq = any(g.goal_type == "mcq" for g in wm.goals)

            diag = {"question": q_text[:80], "expected": expected, "predicted": None,
                    "correct": False, "type": "mcq" if is_mcq else "yesno",
                    "goal_found": bool(wm.goals)}

            if is_mcq:
                counters["total_mcq"] += 1
                result = orchestrator.run_logic(q_item)
                diag["predicted"] = result.answer

                mcq_goal = next((g for g in wm.goals if g.goal_type == "mcq"), None)
                if mcq_goal and any(o.fol_str for o in mcq_goal.options):
                    counters["mcq_canonicalized"] += 1

                if expected and result.answer and result.answer.upper() == expected.upper():
                    counters["mcq_correct"] += 1
                    diag["correct"] = True
            else:
                counters["total_yesno"] += 1
                entailment_goal = next((g for g in wm.goals if g.goal_type == "entailment"), None)
                if entailment_goal:
                    counters["yesno_target_found"] += 1
                    result = orchestrator.run_logic(q_item)
                    diag["predicted"] = result.answer
                    if expected and result.answer and result.answer.lower() == expected.lower():
                        counters["yesno_correct"] += 1
                        diag["correct"] = True

            diagnostics.append(diag)

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
    total_correct = counters['mcq_correct'] + counters['yesno_correct']
    print()
    print(f"  ──── Overall accuracy     : {total_correct}/{n_q} ({total_correct/max(n_q,1)*100:.1f}%)")
    print("══════════════════════════════════════════")

    # Export JSON
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/logic_diagnostics.json", "w") as f:
        json.dump({"counters": counters, "details": diagnostics}, f, indent=2, ensure_ascii=False)
    print(f"  Diagnostics → outputs/logic_diagnostics.json")


if __name__ == "__main__":
    path = os.path.join("data", "Logic_Based_Educational_Queries.json")
    if os.path.exists(path):
        run_benchmark(path)
    else:
        print(f"Dataset not found: {path}")
