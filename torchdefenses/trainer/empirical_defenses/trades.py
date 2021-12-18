import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks import TPGD

from ..advtrainer import AdvTrainer

class TRADES(AdvTrainer):
    r"""
    Adversarial training in 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

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
    def __init__(self, model, eps, alpha, steps, beta):
        super().__init__("TRADES", model)
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
        loss_ce = nn.CrossEntropyLoss()(logits_clean, Y)
        
        logits_adv = self.model(X_adv)
        prob_clean = F.softmax(logits_clean, dim=1)
        log_prob_adv = F.log_softmax(logits_adv, dim=1)
        loss_kl = nn.KLDivLoss(reduction='batchmean')(log_prob_adv, prob_clean)
        
        cost = loss_ce + self.beta * loss_kl
        
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost.item(), loss_ce.item(), loss_kl.item()
