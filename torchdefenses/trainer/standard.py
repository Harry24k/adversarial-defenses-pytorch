import torch
import torch.nn as nn

from .advtrainer import AdvTrainer

r"""
Standard Training Method.

Attributes:
    self.model : model.
    self.device : device where model is.
    self.optimizer : optimizer.
    self.scheduler : scheduler (Automatically updated).
    self.max_epoch : total number of epochs.
    self.max_iter : total number of iterations.
    self.epoch : current epoch starts from 1 (Automatically updated).
    self.iter : current iters starts from 1 (Automatically updated).
        * e.g., is_last_batch = (self.iter == self.max_iter)
    self.record_keys : names of items returned by do_iter.
    
Arguments:
    model (nn.Module): model to train.

"""

class Standard(AdvTrainer):
    def __init__(self, model):
        super().__init__("Standard", model)
        self.record_keys = ["Loss"] # Must be same as the items returned by self._do_iter
    
    # Override Do Iter
    def _do_iter(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        X = images.to(self.device)
        Y = labels.to(self.device)

        pre = self.model(X)
        cost = nn.CrossEntropyLoss()(pre, Y)

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()
        
        return cost.item()