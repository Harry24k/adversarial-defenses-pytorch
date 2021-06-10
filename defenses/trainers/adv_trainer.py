import os
import torch

# from torchhk import Trainer
from .trainer import Trainer
from torchattacks.attack import Attack
from torchattacks import VANILA, FGSM, PGD, GN

r"""
Trainer for Adversarial Training.

Functions:
    self.record_rob : recording standard accuracy and robust accuracy against FGSM, PGD, and GN.

"""

class AdvTrainer(Trainer):
    def __init__(self, name, model):
        super(AdvTrainer, self).__init__(name, model)
        self._flag_record_rob = False
    
    def record_rob(self, train_loader, val_loader, eps, alpha, steps, sigma=0.1, num_limit=1000):
        self.record_keys += ['Clean(Tr)', 'FGSM(Tr)', 'PGD(Tr)', 'GN(Tr)',
                             'Clean(Val)', 'FGSM(Val)', 'PGD(Val)', 'GN(Val)',]
        self.record_atks = [VANILA(self.model), FGSM(self.model, eps=eps),
                            PGD(self.model, eps=eps, alpha=alpha, steps=steps),
                            GN(self.model, sigma=sigma)]
        
        self.train_loader_rob = train_loader
        self.val_loader_rob = val_loader
        self._flag_record_rob = True
        self.num_limit_rob = num_limit
    
    # Update Records
    def _update_record(self, records):
        if self._flag_record_rob:
            adv_list = []
            for atk in self.record_atks:
                acc = get_acc(self.model, atk, self.train_loader_rob,
                              device=self.device, num_limit=self.num_limit_rob)
                adv_list.append(acc)
            for atk in self.record_atks:
                acc = get_acc(self.model, atk, self.val_loader_rob,
                              device=self.device, num_limit=self.num_limit_rob)
                adv_list.append(acc)
            
            self.rm.add([*records,
                         *adv_list,
                         self.optimizer.param_groups[0]['lr']])
        else:
            self.rm.add([*records,
                         self.optimizer.param_groups[0]['lr']])
                         
                
def get_acc(model, atk, data_loader, device='cuda', num_limit=None):
    # Set Cuda or Cpu
    device = torch.device(device)
    model.to(device)
    
    # Set Model to Evaluation Mode
    model.eval()
    
    # Initialize
    correct = 0
    total = 0

    # For all Test Data
    for batch_images, batch_labels in data_loader:

        # Get Batches
        X = batch_images.to(device)
        Y = batch_labels.to(device)
        
        # Forward
        X_adv = atk(X, Y)
        pre = model(X_adv)

        # Calculate Accuracy
        _, pre = torch.max(pre.data, 1)
        total += pre.size(0)
        correct += (pre == Y).sum()
        
        if num_limit is not None:
            if total > num_limit:
                break

    return (100 * float(correct) / total)
