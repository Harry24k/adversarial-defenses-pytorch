import torch
import torch.nn as nn

from torchattacks import FGSM

from .adv_trainer import AdvTrainer

r"""
FGSM Adversarial Training in the paper 'Explaining and harnessing adversarial examples'
[https://arxiv.org/abs/1412.6572]

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

"""

class FGSMAdvTrainer(AdvTrainer):
    def __init__(self, model, eps):
        super(FGSMAdvTrainer, self).__init__("FGSMAdvTrainer", model)
        self.record_keys = ["Loss"] # Must be same as the items returned by self._do_iter
        self.atk = FGSM(model, eps)
    
    def _do_iter(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        X = images.to(self.device)
        Y = labels.to(self.device)

        X_adv = self.atk(X, Y)

        pre = self.model(X_adv)
        cost = nn.CrossEntropyLoss()(pre, Y)

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost.item()
    
    