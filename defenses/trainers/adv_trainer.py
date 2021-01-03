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
    
    def record_rob(self, train_set, test_set, eps, alpha, steps):
        self.record_keys += ['Clean(Tr)', 'FGSM(Tr)', 'PGD(Tr)', 'GN(Tr)',
                             'Clean(Te)', 'FGSM(Te)', 'PGD(Te)', 'GN(Te)',]
        
        self.train_set = train_set
        self.test_set = test_set
        self.record_atks = [VANILA(self.model), FGSM(self.model, eps=eps),
                            PGD(self.model, eps=eps, alpha=alpha, steps=steps),
                            GN(self.model, sigma=0.1)]
        self._flag_record_rob = True
    
    # Update Records
    def _update_record(self, records):
        if self._flag_record_rob:
            adv_list = []
            for atk in self.record_atks:
                adv_list.append([(atk(*self.train_set), self.train_set[1])])
            for atk in self.record_atks:
                adv_list.append([(atk(*self.test_set), self.test_set[1])])
            
            self.rm.add([*records,
                         *[get_acc(self.model, adv_data) for adv_data in adv_list],
                         self.optimizer.param_groups[0]['lr']])
        else:
            self.rm.add([*records,
                         self.optimizer.param_groups[0]['lr']])
                         
                
def get_acc(model, test_loader, device='cuda'):
    # Set Cuda or Cpu
    device = torch.device(device)
    model.to(device)
    
    # Set Model to Evaluation Mode
    model.eval()
    
    # Initialize
    correct = 0
    total = 0

    # For all Test Data
    for batch_images, batch_labels in test_loader:

        # Get Batches
        X = batch_images.to(device)
        Y = batch_labels.to(device)
        
        # Forward
        pre = model(X)

        # Calculate Accuracy
        _, pre = torch.max(pre.data, 1)
        total += pre.size(0)
        correct += (pre == Y).sum()

    return (100 * float(correct) / total)
