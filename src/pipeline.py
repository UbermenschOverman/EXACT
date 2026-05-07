# src/pipeline.py

"""
Pipeline orchestration — supports single-sample and batch modes.
Connects LLM, orchestrator, and all solvers.
"""

import os
from tqdm import tqdm

from src.llm.models import LLMModel, load_llm
from src.llm.generate import LLMGenerator
from src.llm.planner import Planner
from src.llm.translator import Translator
from src.llm.explainer import Explainer

from src.orchestrator import Orchestrator

from src.utils.io import load_json, save_json
from src.utils.logging import setup_logger


class Pipeline:
    def __init__(self, llm=None, config=None):
        self.logger = setup_logger("pipeline")

        # Support config dict (from run.py) or standalone
        if config and isinstance(config, dict):
            self.data_path = config.get("data_path", "data/sample.json")
            self.output_path = config.get("output_path", "outputs/predictions.json")
        else:
            self.data_path = "data/sample.json"
            self.output_path = "outputs/predictions.json"

        # Use provided LLM or load from config
        if llm is None:
            if config and isinstance(config, dict) and "llm" in config:
                llm = load_llm(config["llm"])
            else:
                from src.config import Config
                cfg = Config()
                llm = load_llm(cfg)

        generator = LLMGenerator(llm)

        self.planner = Planner(generator)
        self.translator = Translator(generator)
        self.explainer = Explainer(generator)

        self.orchestrator = Orchestrator(
            self.planner,
            self.translator,
            self.explainer,
            generator=generator,  # pass generator for classifier + premise extraction
        )

    def run(self, sample=None):
        """
        If sample is provided, run on single sample (called from run.py).
        Otherwise, run on full dataset from self.data_path.
        """
        if sample is not None:
            return self.orchestrator.run(sample.question)

        # Batch mode
        data = load_json(self.data_path)

        results = []

        for item in tqdm(data):
            question = item.get("question", "")
            output = self.orchestrator.run(question)
            results.append(output)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        save_json(results, self.output_path)

        self.logger.info(f"Saved results to {self.output_path}")

        return results


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()