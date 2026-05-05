# src/data/collator.py

from typing import List


class Collator:
    def __init__(self, tokenizer=None, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List):
        questions = [x.question for x in batch]

        if self.tokenizer:
            encoded = self.tokenizer(
                questions,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "raw": batch
            }

        return {
            "questions": questions,
            "raw": batch
        }