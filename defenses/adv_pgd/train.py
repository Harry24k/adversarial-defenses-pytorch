import torch
import torch.nn as nn

from torchattacks import PGD

from ..basetrain import BaseTrainer, get_acc

print("Train : Adv PGD")

class Trainer(BaseTrainer):
    r"""
    PGD(Linf) Adversarial Training in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    
    Arguments:
        model (nn.Module): model to train.
        eps (float): strength of the attack or maximum perturbation.
        alpha (float): step size.
        steps (int): nu,ber of steps.
        random_start (bool): using random initialization of delta.
        
    """
    def __init__(self, model, train_sets, test_sets,
                 eps, alpha, steps, random_start):
        super(Trainer, self).__init__(model, train_sets, test_sets)
        self.record_keys = ["Loss", "Acc"]
        self.pgd = PGD(model, eps, alpha, steps, random_start)
    
    def _do_iter(self, images, labels):
        r"""
        Overridden.
        """
        X = images.to(self.device)
        Y = labels.to(self.device)

        X = self.pgd(X, Y)

        pre = self.model(X)
        cost = nn.CrossEntropyLoss()(pre, Y)

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        _, pre = torch.max(pre.data, 1)
        total = pre.size(0)
        correct = (pre == Y).sum()
        cost = cost.item()
        
        return cost, 100*float(correct)/total
    
    