# src/utils/device.py

import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_num_gpus():
    return torch.cuda.device_count()