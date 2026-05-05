# src/data/dataset.py

from typing import List
from .schema import QASample, FOLSample, PhysicsSample
from src.utils.io import load_json


class Dataset:
    def __init__(self, path: str, task_type: str = "qa"):
        self.path = path
        self.task_type = task_type
        self.data = self._load()

    def _load(self):
        raw = load_json(self.path)

        if self.task_type == "fol":
            return [self._parse_fol(x) for x in raw]

        elif self.task_type == "physics":
            return [self._parse_physics(x) for x in raw]

        else:
            return [self._parse_qa(x) for x in raw]

    def _parse_qa(self, x):
        return QASample(
            question=x.get("question", ""),
            answer=x.get("answer"),
            explanation=x.get("explanation")
        )

    def _parse_fol(self, x):
        return FOLSample(
            question=x.get("question", ""),
            answer=x.get("answer"),
            explanation=x.get("explanation"),
            fol=x.get("fol")
        )

    def _parse_physics(self, x):
        return PhysicsSample(
            question=x.get("question", ""),
            answer=x.get("answer"),
            reasoning=x.get("reasoning"),
            formula=x.get("formula")
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]