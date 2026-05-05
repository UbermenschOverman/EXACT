import os
import torch


class Config:
    def __init__(self):
        # ===== MODEL =====
        self.MODEL_NAME_OR_PATH = "/mnt/data2/Ducnguyen/EXACT/models/mistral/Mistral-7B-Instruct-v0.2"

        # ===== GENERATION =====
        self.MAX_NEW_TOKENS = 256
        self.TEMPERATURE = 0.7

        # ===== PATHS =====
        self.DATA_PATH = "data/sample.json"
        self.OUTPUT_PATH = "outputs/"
        self.CACHE_PATH = "cache/"

        # ===== RUNTIME =====
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"