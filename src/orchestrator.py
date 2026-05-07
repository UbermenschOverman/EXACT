# src/orchestrator.py

"""
Central orchestrator — routes questions to appropriate solver (physics/logic/general)
and produces unified ReasoningOutput.
"""

from src.llm.planner import Planner
from src.llm.translator import Translator
from src.llm.explainer import Explainer
from src.llm.classifier import QuestionClassifier

from src.reasoning.parser import FOLParser
from src.reasoning.logic_solver import LogicSolver
from src.reasoning.physics_solver import PhysicsSolver
from src.reasoning.proof import ProofGenerator
from src.reasoning.verifier import Verifier

from src.data.schema import ReasoningOutput
from src.utils.json_utils import safe_json_load


class Orchestrator:
    def __init__(
        self,
        planner: Planner,
        translator: Translator,
        explainer: Explainer,
        generator=None,
        max_retries: int = 2
    ):
        self.planner = planner
        self.translator = translator
        self.explainer = explainer
        self.generator = generator

        self.classifier = QuestionClassifier()
        self.parser = FOLParser()
        self.logic_solver = LogicSolver()
        self.physics_solver = PhysicsSolver()
        self.proof = ProofGenerator()
        self.verifier = Verifier()

        self.max_retries = max_retries

    def run(self, question: str) -> dict:
        """
        Main entry point. Classifies question and routes to appropriate solver.
        Returns dict (serializable ReasoningOutput).
        """
        # 1. Classify question type
        q_type = self.classifier.classify(question, self.generator)
        print(f"[Orchestrator] Question type: {q_type}")

        error_msg = "No attempts made"

        for attempt in range(self.max_retries + 1):
            try:
                if q_type == "physics":
                    result = self._run_physics(question, attempt)
                elif q_type == "logic":
                    result = self._run_logic(question, attempt)
                else:
                    result = self._run_general(question, q_type, attempt)

                # Verify result
                verification = self.verifier.verify(result)
                result["verification"] = verification
                result["valid"] = verification.get("valid", False)

                # If valid or partially valid, generate explanation
                if result.get("valid") or result.get("confidence", 0) > 0.3:
                    trace_text = self._format_trace(result.get("reasoning_trace", []))
                    explanation = self.explainer.explain(
                        question,
                        reasoning=trace_text,
                        answer=result.get("answer", "")
                    )
                    result["explanation"] = explanation
                    result["attempt"] = attempt
                    result["question_type"] = q_type
                    return result

            except Exception as e:
                error_msg = str(e)
                print(f"[Orchestrator] Attempt {attempt} failed: {error_msg}")

        # Fallback
        return ReasoningOutput(
            question=question,
            answer="UNKNOWN",
            question_type=q_type,
            error=f"Failed after retries. Last error: {error_msg}",
        ).to_dict()

    def _run_physics(self, question: str, attempt: int) -> dict:
        """
        Physics pipeline: extract vars → detect formula → SymPy solve → trace.
        Deterministic — no LLM for computation.
        """
        result = self.physics_solver.solve_question(question)

        # Generate proof from trace
        proof_steps = self.proof.generate(result.get("reasoning_trace", []))

        return {
            "question": question,
            "answer": result["answer"],
            "explanation": "",
            "reasoning_trace": result.get("reasoning_trace", []),
            "used_premises": result.get("used_premises", []),
            "confidence": result.get("confidence", 0.0),
            "question_type": "physics",
            "valid": result.get("confidence", 0) > 0.5,
            "proof": proof_steps,
            "formula_used": result.get("formula_used", ""),
            "numeric_value": result.get("numeric_value"),
            "unit": result.get("unit", ""),
        }

    def _run_logic(self, question: str, attempt: int) -> dict:
        """
        Logic pipeline: LLM extracts premises → build graph → forward chain → prove.
        """
        # Use LLM to extract premises
        from src.llm.prompt import build_premise_extraction_prompt
        prompt = build_premise_extraction_prompt(question)
        raw_output = self.generator.run(prompt, temperature=0.2) if self.generator else ""

        parsed = safe_json_load(raw_output)

        if parsed:
            premises = parsed.get("premises", [])
            rules = parsed.get("rules", [])
            target = parsed.get("target", None)
        else:
            # Fallback: use FOL translator
            fol = self.translator.to_fol(question)
            extracted = self.parser.extract_premises_and_rules(fol)
            premises = extracted["premises"]
            rules = extracted["rules"]
            target = None

        # Solve with logic solver
        result = self.logic_solver.solve(premises, rules, target)

        # Generate proof
        proof_steps = self.proof.generate(result.get("reasoning_trace", []))

        return {
            "question": question,
            "answer": result["answer"],
            "explanation": "",
            "reasoning_trace": result.get("reasoning_trace", []),
            "used_premises": result.get("used_premises", []),
            "confidence": result.get("confidence", 0.0),
            "question_type": "logic",
            "valid": result.get("confidence", 0) > 0.3,
            "proof": proof_steps,
            "contradiction": result.get("contradiction"),
        }

    def _run_general(self, question: str, q_type: str, attempt: int) -> dict:
        """
        General pipeline: LLM planner + translator + existing flow.
        Used for definitions and unknown question types.
        """
        # Plan steps
        steps = self.planner.plan(question)

        # Translate to FOL (best effort)
        fol = self.translator.to_fol(question)

        # Parse
        parsed = self.parser.parse(fol)

        trace = [
            {"step": 1, "action": "plan", "detail": f"Steps: {steps}"},
            {"step": 2, "action": "translate", "detail": f"FOL: {fol}"},
            {"step": 3, "action": "parse", "detail": f"Parsed: {parsed}"},
        ]

        # For definitions, the LLM explanation IS the answer
        answer = fol if q_type == "definition" else str(parsed.get("value", fol))

        return {
            "question": question,
            "answer": answer,
            "explanation": "",
            "reasoning_trace": trace,
            "used_premises": [f"Step: {s}" for s in steps],
            "confidence": 0.5,
            "question_type": q_type,
            "valid": True,
            "steps": steps,
            "fol": fol,
        }

    def _format_trace(self, trace: list) -> str:
        """Format reasoning trace as human-readable text for LLM explainer."""
        lines = []
        for entry in trace:
            detail = entry.get("detail", entry.get("text", ""))
            lines.append(f"[{entry.get('action', '?')}] {detail}")
        return "\n".join(lines)