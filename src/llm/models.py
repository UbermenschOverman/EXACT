import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMWrapper:
    def __init__(self, model_path: str):
        self.model_path = os.path.abspath(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load()

    def _load(self):
        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        print("Loading model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=True
        )

        if not torch.cuda.is_available():
            self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens=256, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_llm(config):
    if isinstance(config, dict):
        model_path = config.get("model_path", config.get("MODEL_NAME_OR_PATH", ""))
    else:
        model_path = getattr(config, "MODEL_NAME_OR_PATH", "")
    return LLMWrapper(model_path)


# Alias for backward compatibility
LLMModel = LLMWrapper