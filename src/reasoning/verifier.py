# src/reasoning/verifier.py

class Verifier:
    def verify(self, answer: str, reasoning: str):
        """
        Basic consistency check
        """

        if not answer:
            return False

        if answer.lower() in reasoning.lower():
            return True

        return False