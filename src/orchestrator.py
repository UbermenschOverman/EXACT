# src/orchestrator.py

from src.llm.planner import Planner
from src.llm.translator import Translator
from src.llm.explainer import Explainer

from src.reasoning.parser import FOLParser
from src.reasoning.solver import Solver
from src.reasoning.proof import ProofGenerator
from src.reasoning.verifier import Verifier


class Orchestrator:
    def __init__(
        self,
        planner: Planner,
        translator: Translator,
        explainer: Explainer,
        max_retries: int = 2
    ):
        self.planner = planner
        self.translator = translator
        self.explainer = explainer

        self.parser = FOLParser()
        self.solver = Solver()
        self.proof = ProofGenerator()
        self.verifier = Verifier()

        self.max_retries = max_retries

    def run(self, question: str):
        """
        Main reasoning pipeline
        """

        for attempt in range(self.max_retries + 1):

            try:
                # 1. PLAN
                steps = self.planner.plan(question)

                # 2. TRANSLATE → FOL
                fol = self.translator.to_fol(question)

                # 3. PARSE
                parsed = self.parser.parse(fol)

                # 4. SOLVE
                result = self.solver.solve(parsed)

                # 5. PROOF TRACE
                proof_steps = self.proof.generate(parsed, result)

                # 6. VERIFY
                is_valid = self.verifier.verify(str(result), str(proof_steps))

                if is_valid:
                    explanation = self.explainer.explain(
                        question,
                        reasoning="\n".join(proof_steps),
                        answer=str(result)
                    )

                    return {
                        "question": question,
                        "steps": steps,
                        "fol": fol,
                        "parsed": parsed,
                        "answer": str(result),
                        "proof": proof_steps,
                        "explanation": explanation,
                        "valid": True,
                        "attempt": attempt
                    }

            except Exception as e:
                error_msg = str(e)

        # fallback nếu fail toàn bộ
        return {
            "question": question,
            "answer": "UNKNOWN",
            "valid": False,
            "error": "Failed after retries"
        }