import os
import torch

from ._trainer import Trainer

r"""
Base class for adversarial trainers.

Functions:
    self.record_rob : function for recording standard accuracy and robust accuracy against FGSM, PGD, and GN.

"""

class AdvTrainer(Trainer):
    def __init__(self, name, model):
        super(AdvTrainer, self).__init__(name, model)
        self._flag_record_rob = False
    
    def record_rob(self, train_loader, val_loader, eps, alpha, steps, std=0.1, n_limit=1000):
        self.record_keys += ['Clean(Tr)', 'FGSM(Tr)', 'PGD(Tr)', 'GN(Tr)',
                             'Clean(Val)', 'FGSM(Val)', 'PGD(Val)', 'GN(Val)',]    
        self._flag_record_rob = True    
        self._train_loader_rob = self.get_sample_loader(train_loader, n_limit)
        self._val_loader_rob = val_loader
        self._eps_rob = eps
        self._alpha_rob = alpha
        self._steps_rob = steps
        self._std_rob = std
    
    # Update Records
    def _update_record(self, records):
        if self._flag_record_rob:
            rob_records = []
            for loader in [self._train_loader_rob, self._val_loader_rob]:
                rob_records.append(self.model.eval_accuracy(loader))
                rob_records.append(self.model.eval_rob_accuracy_fgsm(loader,
                                                                     eps=self._eps_rob, verbose=False))
                rob_records.append(self.model.eval_rob_accuracy_pgd(loader,
                                                                    eps=self._eps_rob,
                                                                    alpha=self._alpha_rob,
                                                                    steps=self._steps_rob,
                                                                    verbose=False))
                rob_records.append(self.model.eval_rob_accuracy_gn(loader,
                                                                   std=self._std_rob,
                                                                   verbose=False))
            
            self.rm.add([*records,
                         *rob_records,
                         self.optimizer.param_groups[0]['lr']])
        else:
            self.rm.add([*records,
                         self.optimizer.param_groups[0]['lr']])

    def get_sample_loader(self, given_loader, n_limit):
        final_loader = []
        num = 0
        for item in given_loader:
            final_loader.append(item)
            if isinstance(item, tuple) or isinstance(item, list):
                batch_size = len(item[0])
            else:
                batch_size = len(item)
            num += batch_size
            if num >= n_limit:
                break
        return final_loader