# Modified from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/wrn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
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
    
    
class Identity(nn.Module):
    def __init__(self) :
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
