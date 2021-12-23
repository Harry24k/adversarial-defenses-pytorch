import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks import PGD, TPGD

from ..advtrainer import AdvTrainer

class BAT(AdvTrainer):
    r"""
    Adversarial training in 'Bridged Adversarial Training'
    [https://arxiv.org/abs/2108.11135]

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
        m (int): number of bridges.
        loss (str): inner maximization loss. [Options: 'ce', 'kl'] (Default: 'ce')
    """
    def __init__(self, model, eps, alpha, steps, beta, m, loss='ce'):
        super().__init__("BAT", model)
        self.record_keys = ["Loss", "CELoss", "KLLoss"] # Must be same as the items returned by self._do_iter
        if loss == 'ce':
            self.atk = PGD(model, eps, alpha, steps)
        elif loss == 'kl':
            self.atk = TPGD(model, eps, alpha, steps)
        else:
            raise ValueError(type + " is not a valid type. [Options: 'ce', 'kl']")
        self.beta = beta
        self.m = m
    
    def _do_iter(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        X = images.to(self.device)
        Y = labels.to(self.device)

        logits_clean = self.model(X)
        prob_clean = F.softmax(logits_clean, dim=1)

        loss_natural = nn.CrossEntropyLoss()(logits_clean, Y)
                
        X_adv = self.atk(X, Y)
        pert = (X_adv-X).detach()

        loss_robust = 0
        prob_pre = prob_clean
        for i in range(self.m) :
            logits_adv = self.model(X + pert*(i+1)/self.m)
            log_prob_adv = F.log_softmax(logits_adv, dim=1)
            #cost += nn.CrossEntropyLoss()(pre, Y)/(i+1)
            loss_robust += nn.KLDivLoss(reduction='batchmean')(log_prob_adv, prob_pre)
            prob_pre = F.softmax(logits_adv, dim=1)

        cost = loss_natural + self.beta*loss_robust
            
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost.item(), loss_natural.item(), loss_robust.item()
