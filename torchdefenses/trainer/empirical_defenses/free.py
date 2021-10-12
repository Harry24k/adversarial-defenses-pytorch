import torch
import torch.nn as nn

from ..advtrainer import AdvTrainer

class Free(AdvTrainer):
    r"""
    Free adversarial training in 'Adversarial Training for Free!'
    [https://arxiv.org/pdf/1904.12843.pdf]

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
        eps (float): strength of the attack or maximum perturbation.
        m (int) : hop steps.
    """
    def __init__(self, model, eps, m):
        super().__init__("Free", model)
        self.record_keys = ["CALoss"] # Must be same as the items returned by self._do_iter
        self.eps = eps
        self.m = m
        self.m_accumulated = 0
        self.repeated_data = None
        self.delta = 0
    
    def _do_iter(self, train_data):
        r"""
        Overridden.
        """
        if self.m_accumulated % self.m == 0:
            self.repeated_data = train_data
        self.m_accumulated += 1
            
        images, labels = self.repeated_data
        X = images.clone().detach().to(self.device)
        Y = labels.clone().detach().to(self.device)

        X.requires_grad = True

        logits_adv = self.model(X + self.delta)
        cost = nn.CrossEntropyLoss()(logits_adv, Y)
        
        self.optimizer.zero_grad()
        cost.backward()

        grad = X.grad.detach()

        self.optimizer.step()

        self.delta = self.delta + self.eps*grad.sign()
        self.delta = torch.clamp(self.delta.data, -self.eps,  self.eps)

        return cost.item()
