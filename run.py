# run.py

import argparse
from src.utils.config_loader import load_config
from src.utils.logging import setup_logger
from src.utils.io import save_json
from src.utils.cache import Cache

from src.data.dataset import Dataset
from src.llm.models import load_llm
from src.pipeline import Pipeline
from src.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # --- load config ---
    config = load_config(args.config)

    # --- logger ---
    logger = setup_logger("run")

    # --- cache ---
    cache = Cache(config.get("cache_dir", "cache"))

    # --- dataset ---
    dataset = Dataset(
        config["data_path"],
        task_type=config.get("task_type", "qa")
    )

    # --- model ---
    llm = load_llm(config["llm"])

    # --- pipeline ---
    pipeline = Pipeline(llm, config)

    # --- evaluator ---
    evaluator = Evaluator()

    predictions = []
    references = []

    logger.info(f"Running on {len(dataset)} samples")

    for i, sample in enumerate(dataset):

        cache_key = {
            "question": sample.question,
            "config": config["llm"]
        }

        cached = cache.get(cache_key)

        if cached:
            output = cached
        else:
            output = pipeline.run(sample)
            cache.set(cache_key, output)

        predictions.append(output)

        references.append({
            "answer": sample.answer,
            "explanation": sample.explanation
        })

        if i % 10 == 0:
            logger.info(f"Processed {i}/{len(dataset)}")

    # --- evaluate ---
    results = evaluator.evaluate(predictions, references)

    logger.info(f"Results: {results}")

    # --- save ---
    save_json(predictions, config["output_path"])
    save_json(results, config["output_path"] + ".metrics.json")


if __name__ == "__main__":
    main()