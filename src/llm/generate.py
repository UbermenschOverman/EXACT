# src/llm/generate.py

import hashlib
from src.utils.cache import Cache


class LLMGenerator:
    def __init__(self, model, use_cache=True):
        self.model = model
        self.cache = Cache() if use_cache else None

    def _hash(self, prompt: str):
        return hashlib.md5(prompt.encode()).hexdigest()

    def run(self, prompt: str, **kwargs):
        key = self._hash(prompt)

        if self.cache:
            cached = self.cache.get(key)
            if cached:
                return cached

        raw_output = self.model.generate(prompt, **kwargs)
        cleaned = self._clean_output(raw_output, prompt)

        if self.cache:
            self.cache.set(key, cleaned)

        return cleaned

    def _clean_output(self, text: str, prompt: str):
        # remove prompt echo
        if text.startswith(prompt):
            text = text[len(prompt):]

        return text.strip()