import numpy as np
import random
import torch
import torch.nn as nn
import copy

# FIX Randomness (https://hoya012.github.io/)
def fix_randomness(random_seed=0):
    random_seed = 617
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    print("Random Seed:",random_seed)
    
class Arguments:
    def __init__(self):
        pass