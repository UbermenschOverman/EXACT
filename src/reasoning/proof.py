# src/reasoning/proof.py

class ProofGenerator:
    def generate(self, parsed_fol, result):
        steps = []

        if parsed_fol["type"] == "implication":
            steps.append(f"Given premise: {parsed_fol['premise']}")
            steps.append(f"If premise holds, then: {parsed_fol['conclusion']}")

        steps.append(f"Result: {result}")

        return steps