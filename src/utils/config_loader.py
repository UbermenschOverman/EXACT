# src/utils/config_loader.py

import yaml
from typing import Dict


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_config(base: Dict, override: Dict) -> Dict:
    """
    recursive merge
    """

    result = base.copy()

    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = v

    return result