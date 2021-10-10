# Modified from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/wrn.py
import math
import torch
from torch.nn import Module
import torch.nn.functional as F

class Normalize(Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        
    def __str__(self):
        info = self.__dict__
        return "Normalize(" + 'mean={}, std={}'.format(self.mean, self.std) + ")"
    
    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std
