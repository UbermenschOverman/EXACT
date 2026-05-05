# src/utils/io.py

import json
import os
from typing import Any


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def append_jsonl(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(path: str):
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)