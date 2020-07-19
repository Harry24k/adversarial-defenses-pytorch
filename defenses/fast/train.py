import torch
import torch.nn as nn

from torchattacks import FFGSM

from ..basetrain import BaseTrainer, get_acc

print("Train : Fast")

class Trainer(BaseTrainer):
    r"""
    'Fast is better than free: Revisiting adversarial training'
    [https://arxiv.org/abs/2001.03994]
    
    Arguments:
        model (nn.Module): model to train.
        eps (float): strength of the attack or maximum perturbation.
        alpha (float): alpha in the paper.
        
    """
    def __init__(self, model, train_sets, test_sets,
                 eps, alpha):
        super(Trainer, self).__init__(model, train_sets, test_sets)
        self.record_keys = ["Loss", "Acc"]
        self.ffgsm = FFGSM(model, eps, alpha)
    
    def _do_iter(self, images, labels):
        r"""
        Overridden.
        """
        X = images.to(self.device)
        Y = labels.to(self.device)

        X = self.ffgsm(X, Y)

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
    
    