from src.config import Config
from src.llm.models import load_llm


def main():
    config = Config()
    llm = load_llm(config)

    prompt = "Explain why the sky is blue."
    out = llm.generate(prompt)

    print("\n=== OUTPUT ===")
    print(out)


if __name__ == "__main__":
    main()