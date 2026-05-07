# src/utils/cache.py

"""
Namespaced JSON file cache.
Supports separate cache directories for different components.
Only caches successful results by default.
"""

import hashlib
import json
import os
from typing import Any, Optional


class Cache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_namespace_dir(self, namespace: str) -> str:
        """Get directory for a cache namespace."""
        ns_dir = os.path.join(self.cache_dir, namespace)
        os.makedirs(ns_dir, exist_ok=True)
        return ns_dir

    def _hash(self, key: Any) -> str:
        if isinstance(key, str):
            key_str = key
        else:
            key_str = json.dumps(key, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: Any, namespace: str = "default") -> Optional[Any]:
        """Get cached value. Returns None if not found."""
        h = self._hash(key)
        ns_dir = self._get_namespace_dir(namespace)
        path = os.path.join(ns_dir, f"{h}.json")

        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def set(self, key: Any, value: Any, namespace: str = "default") -> None:
        """Cache a value. Does NOT cache if value indicates failure."""
        # Don't cache failures
        if isinstance(value, dict):
            if value.get("answer") == "UNKNOWN" or value.get("error"):
                return

        h = self._hash(key)
        ns_dir = self._get_namespace_dir(namespace)
        path = os.path.join(ns_dir, f"{h}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)

    def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear cache. If namespace is given, clear only that namespace.
        Returns number of files deleted.
        """
        import shutil

        count = 0

        if namespace:
            ns_dir = os.path.join(self.cache_dir, namespace)
            if os.path.exists(ns_dir):
                for f in os.listdir(ns_dir):
                    os.remove(os.path.join(ns_dir, f))
                    count += 1
        else:
            if os.path.exists(self.cache_dir):
                for item in os.listdir(self.cache_dir):
                    item_path = os.path.join(self.cache_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        count += 1
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                        count += 1

        return count

    def exists(self, key: Any, namespace: str = "default") -> bool:
        """Check if a key exists in cache."""
        h = self._hash(key)
        ns_dir = os.path.join(self.cache_dir, namespace)
        path = os.path.join(ns_dir, f"{h}.json")
        return os.path.exists(path)