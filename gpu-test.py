import torch

def check_mps_available():
    if torch.backends.mps.is_available():
        print("mps available")
    else:
        print("mps not available")

check_mps_available()