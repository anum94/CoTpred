import torch
from torch import device


def get_device() -> str:
    # set the device
    if torch.cuda.is_available():
        print("CUDA AVAILABLE....")
        torch.cuda.empty_cache()
        return "cuda"
    else:
        return "cpu"

device = get_device()