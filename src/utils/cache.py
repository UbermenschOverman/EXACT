# src/utils/cache.py

import hashlib
import json
import os
from typing import Any


class Cache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _hash(self, key: Any) -> str:
        key_str = json.dumps(key, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: Any):
        h = self._hash(key)
        path = os.path.join(self.cache_dir, f"{h}.json")

        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def set(self, key: Any, value: Any):
        h = self._hash(key)
        path = os.path.join(self.cache_dir, f"{h}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)