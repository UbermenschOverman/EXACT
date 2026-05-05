# src/llm/test_llm.py

from src.llm.models import LLMModel, load_llm
from src.llm.generate import LLMGenerator
from src.llm.planner import Planner
from src.llm.translator import Translator
from src.llm.explainer import Explainer


def main():
    model = LLMModel("mistralai/Mistral-7B-Instruct-v0.2")
    generator = LLMGenerator(model)

    planner = Planner(generator)
    translator = Translator(generator)
    explainer = Explainer(generator)

    question = "If a student fails more than 2 subjects, can they graduate?"

    steps = planner.plan(question)
    fol = translator.to_fol(question)

    explanation = explainer.explain(
        question,
        reasoning=str(steps) + "\n" + fol,
        answer="No"
    )

    print("STEPS:", steps)
    print("FOL:", fol)
    print("EXPLANATION:", explanation)


if __name__ == "__main__":
    main()