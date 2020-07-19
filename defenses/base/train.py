import torch
import torch.nn as nn

from ..basetrain import BaseTrainer, get_acc

print("Train : Base")

class Trainer(BaseTrainer):
    r"""
    Arguments:
        model (nn.Module): model to train.
        
    """
    def __init__(self, model, train_sets, test_sets):
        super(Trainer, self).__init__(model, train_sets, test_sets)
        self.record_keys = ["Loss", "Acc"]
    
    def _do_iter(self, images, labels):
        r"""
        Overridden.
        """
        X = images.to(self.device)
        Y = labels.to(self.device)

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
    
    