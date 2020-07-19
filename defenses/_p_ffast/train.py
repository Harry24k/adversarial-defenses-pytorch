import torch
import torch.nn as nn

from torchattacks import FFGSM

from ..basetrain import BaseTrainer, get_acc

print("Train : Proj_Fast FFGSM Multi Inference")

class Trainer(BaseTrainer):
    r"""
    Adversarial Training with FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]
    
    Arguments:
        model (nn.Module): model to train.
        eps (float): strength of the attack or maximum perturbation.
        
    """
    def __init__(self, model, train_sets, test_sets, eps, alpha, m):
        super(Trainer, self).__init__(model, train_sets, test_sets)
        self.record_keys = ["Loss", "Acc"]
        self.ffgsm = FFGSM(model, eps, alpha)
        self.m = m
    
    def _do_iter(self, images, labels):
        r"""
        Overridden.
        """
        X = images.to(self.device)
        Y = labels.to(self.device)

        X_adv = self.ffgsm(X, Y)
        pert = (X_adv-X).detach()
        
        X_new = images.clone().detach().to(self.device)
        idxs = torch.arange(len(X_new)).to(self.device)
        
        for _ in range(self.m) :
            pre = self.model(X_new[idxs]).detach()
            _, pre = torch.max(pre.data, 1)
            correct = (pre == Y[idxs])

            right_mask = torch.masked_select(idxs, correct)
            
            if len(right_mask) == 0 :
                break
            
            # 맞은 걸 늘려서 틀릴 때 학습함
            X_new[right_mask] = X_new[right_mask] + pert[right_mask]/self.m
            
            # 근데 사실 틀리기 바로 전의 것만 학습할 수도 있지 않나?
            
            idxs = right_mask
            
            # 한꺼번에 Forward 할 필요는 없지??
            # Step By Step으로 걸러내면서 학습해도 되려나?
        
        pre = self.model(X_new.detach().to(self.device))
        cost = nn.CrossEntropyLoss()(pre, Y)

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        _, pre = torch.max(pre.data, 1)
        total = pre.size(0)
        correct = (pre == Y).sum()
        cost = cost.item()
        
        return cost, 100*float(correct)/total
    
    