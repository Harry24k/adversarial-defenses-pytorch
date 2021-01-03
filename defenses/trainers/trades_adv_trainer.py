import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks import TPGD

from .adv_trainer import AdvTrainer

r"""
'Theoretically Principled Trade-off between Robustness and Accuracy'
[https://arxiv.org/abs/1901.08573]
[https://github.com/yaodongyu/TRADES]

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
    alpha (float): step size.
    steps (int): number of steps.
    beta (float): trade-off regularization parameter.

"""

class TRADESAdvTrainer(AdvTrainer):
    def __init__(self, model, eps, alpha, steps, beta):
        super(TRADESAdvTrainer, self).__init__("TRADESAdvTrainer", model)
        self.record_keys = ["Loss", "CELoss", "KLLoss"] # Must be same as the items returned by self._do_iter
        self.atk = TPGD(model, eps, alpha, steps)
        self.beta = beta
    
    def _do_iter(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        X = images.to(self.device)
        Y = labels.to(self.device)

        X_adv = self.atk(X)

        logits_clean = self.model(X)
        logits_adv = self.model(X_adv)
        
        prob_clean = F.softmax(logits_clean, dim=1)
        log_prob_adv = F.log_softmax(logits_adv, dim=1)
        
        loss_natural = nn.CrossEntropyLoss()(logits_clean, Y)
        loss_robust = nn.KLDivLoss(reduction='batchmean')(log_prob_adv, prob_clean)
        
        cost = loss_natural + self.beta * loss_robust
        
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost.item(), loss_natural.item(), loss_robust.item()
    
    