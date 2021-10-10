import torch
from torch.nn import Module
from .normalize import Normalize

class RobModel(Module):
    r"""
    Wrapper class for PyTorch models.
    """
    def __init__(self, model, num_labels, normalize=None):
        super().__init__()
        self.model = model
        assert isinstance(num_labels, int)
        self.register_buffer('num_labels', torch.tensor(num_labels))        
        if normalize:
            self.norm = Normalize(normalize['mean'], normalize['std'])
        
    def forward(self, x):
        norm_x = self.norm(x)
        out = self.model(norm_x)
        return out
