# utils.py
import torch
import random
import numpy as np

def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy random
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (if you use CUDA)
    torch.backends.cudnn.deterministic = True  # For deterministic behavior with cudnn
    torch.backends.cudnn.benchmark = False  # This ensures reproducibility on GPUs with varying architectures
