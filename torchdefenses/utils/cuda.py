import copy
import random
import numpy as np

import torch
import torch.nn as nn

# Modified from https://hoya012.github.io/
def fix_randomness(random_seed=0):
    random_seed = 617
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    print("Fixed Random Seed:",random_seed)
    
def fix_gpu(n_gpu):
    torch.cuda.set_device(n_gpu)
