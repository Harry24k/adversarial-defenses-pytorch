import torch
import torch.nn as nn

from torchattacks import FGSM

from ..basetrain import BaseTrainer, get_acc

print("Train : Adv FSGM")

class Trainer(BaseTrainer):
    r"""
    Adversarial Training with FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]
    
    Arguments:
        model (nn.Module): model to train.
        eps (float): strength of the attack or maximum perturbation.
        
    """
    def __init__(self, model, train_sets, test_sets, eps):
        super(Trainer, self).__init__(model, train_sets, test_sets)
        self.record_keys = ["Loss", "Acc"]
        self.fgsm = FGSM(model, eps)
    
    def _do_iter(self, images, labels):
        r"""
        Overridden.
        """
        X = images.to(self.device)
        Y = labels.to(self.device)

        X = self.fgsm(X, Y)

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
    
    