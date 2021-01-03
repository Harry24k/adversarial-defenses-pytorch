import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks import PGD

from .adv_trainer import AdvTrainer

r"""
'Improving Adversarial Robustness Requires Revisiting Misclassified Examples'
[https://openreview.net/forum?id=rklOg6EFwS]
[https://github.com/YisenWang/MART]

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
    random_start (bool): using random initialization of delta.
"""

class MARTAdvTrainer(AdvTrainer):
    def __init__(self, model, eps, alpha, steps, beta, random_start=False):
        super(MARTAdvTrainer, self).__init__("MARTAdvTrainer", model)
        self.record_keys = ["Loss", "RBLoss", "KLLoss"] # Must be same as the items returned by self._do_iter
        self.atk = PGD(model, eps, alpha, steps, random_start)
        self.beta = beta
    
    def _do_iter(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        X = images.to(self.device)
        Y = labels.to(self.device)

        X_adv = self.atk(X, Y)
            
        logits_clean = self.model(X)
        logits_adv = self.model(X_adv)
        
        prob_adv = F.softmax(logits_adv, dim=1)
        
        tmp1 = torch.argsort(prob_adv, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == Y, tmp1[:, -2], tmp1[:, -1])
        
        loss_adv = F.cross_entropy(logits_adv, Y) + F.nll_loss(torch.log(1.0001 - prob_adv + 1e-12), new_y)

        prob_clean = F.softmax(logits_clean, dim=1)
        true_probs = torch.gather(prob_clean, 1, (Y.unsqueeze(1)).long()).squeeze()
        
        loss_robust = (1.0 / len(X)) * torch.sum(
            torch.sum(nn.KLDivLoss(reduction='none')(torch.log(prob_adv + 1e-12), prob_clean), dim=1) * (1.0000001 - true_probs))

        cost = loss_adv + self.beta * loss_robust
        
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost.item(), loss_adv.item(), loss_robust.item()
    
    