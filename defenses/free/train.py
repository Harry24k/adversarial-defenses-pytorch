import torch
import torch.nn as nn

from ..basetrain import BaseTrainer, get_acc

print("Train : Adv PGD")

class Trainer(BaseTrainer):
    r"""
    PGD(Linf) Adversarial Training in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    
    Arguments:
        model (nn.Module): model to train.
        eps (float): strength of the attack or maximum perturbation.
        m (int) : hop steps.
        
    """
    def __init__(self, model, train_sets, test_sets,
                 eps, m):
        super(Trainer, self).__init__(model, train_sets, test_sets)
        self.record_keys = ["Loss", "Acc"]
        self.eps = eps
        self.m = m
        self.delta = 0
    
    def _do_iter(self, images, labels):
        r"""
        Overridden.
        """
        for _ in range(self.m) :
            X = images.to(self.device)
            Y = labels.to(self.device)

            X.requires_grad = True

            pre = self.model(X + self.delta)

            cost = nn.CrossEntropyLoss()(pre, Y)

            self.optimizer.zero_grad()
            cost.backward()

            g_adv = X.grad.data

            self.optimizer.step()

            self.delta = self.delta + self.eps*g_adv.sign()
            self.delta = torch.clamp(self.delta.data, -self.eps,  self.eps)
            
        _, pre = torch.max(pre.data, 1)
        total = pre.size(0)
        correct = (pre == Y).sum()
        cost = cost.item()

        return cost, 100*float(correct)/total