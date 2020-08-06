import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks import TPGD

from ..basetrain import BaseTrainer, get_acc

print("Train : TRADES")

class Trainer(BaseTrainer):
    r"""
    'Theoretically Principled Trade-off between Robustness and Accuracy'
    [https://arxiv.org/1901.08573]
    [https://github.com/yaodongyu/TRADES]
    
    Arguments:
        model (nn.Module): model to train.
        eps (float): strength of the attack or maximum perturbation.
        alpha (float): alpha in the paper.
        iters (int): step size.
        beta (float): trade-off regularization parameter.
    """
    def __init__(self, model, train_sets, test_sets,
                 eps, alpha, steps, beta):
        super(Trainer, self).__init__(model, train_sets, test_sets)
        self.record_keys = ["Loss_Nat", "Loss_Rob", "Acc"]
        self.tpgd = TPGD(model, eps, alpha, steps)
        self.beta = beta
    
    def _do_iter(self, images, labels):
        r"""
        Overridden.
        """
        X = images.to(self.device)
        Y = labels.to(self.device)

        X_adv = self.tpgd(X)

        logits_clean = self.model(X)
        logits_adv = self.model(X_adv)
        
        log_prob_adv = F.log_softmax(logits_adv, dim=1)
        prob_clean = F.softmax(logits_clean, dim=1)
        
        loss_natural = nn.CrossEntropyLoss()(logits_clean, Y)
        loss_robust = nn.KLDivLoss(reduction='batchmean')(log_prob_adv, prob_clean)
        
        cost = loss_natural + self.beta * loss_robust
        
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        _, pre = torch.max(logits_clean.data, 1)
        total = pre.size(0)
        correct = (pre == Y).sum()
        
        return loss_natural.item(), loss_robust.item(), 100*float(correct)/total
    
    