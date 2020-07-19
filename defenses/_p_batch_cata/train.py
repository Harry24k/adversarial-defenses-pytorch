import torch
import torch.nn as nn

from torchattacks import FFGSM

from ..basetrain import BaseTrainer, get_acc

print("Train : Fast Batch Cata")

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
        self.data_his = {}
    
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

        if (self.epoch+1) == 22 :
            if (self.iter+1) >= 148 and (self.iter+1) <=200 :
                self.data_his[(epoch+1, i+1)] = (images, labels, X.cpu().data)
                torch.save(self.model.cpu().state_dict(),
                           "./_models/_p_batch_cata/"+str(self.epoch+1).zfill(3)+"_"+str(self.iter+1).zfill(3)+".pth")
            if (self.iter+1) > 200 :
                torch.save(self.data_his, './_models/_p_batch_cata'+"data.pt")
        
        
        return cost, 100*float(correct)/total
    
    