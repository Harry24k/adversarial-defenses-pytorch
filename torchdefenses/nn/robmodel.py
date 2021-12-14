import torch
import torch.nn as nn

from torchattacks import FGSM, PGD, GN, AutoAttack, MultiAttack

from .modules.normalize import Normalize
from .functional import get_acc

class RobModel(nn.Module):
    r"""
    Wrapper class for PyTorch models.
    """
    def __init__(self, model, n_classes, normalize=None):
        super().__init__()
        assert isinstance(n_classes, int)
        self.model = model
        device = next(model.parameters()).device
        self.register_buffer('n_classes', torch.tensor(n_classes))        
        if normalize:
            self.norm = Normalize(normalize['mean'], normalize['std'])
            self.norm = self.norm.to(device)
            self.model = nn.Sequential(self.norm, model)
            
    def forward(self, x):
        out = self.model(x)
        return out

    # Evaluation Robustness
    def eval_accuracy(self, data_loader, n_limit=None):
        return get_acc(self, data_loader, n_limit=n_limit)
        
    def eval_rob_accuracy(self, data_loader, atk, n_limit=None):
        return get_acc(self, data_loader, n_limit=n_limit, atk=atk)
        
    def eval_rob_accuracy_gn(self, data_loader, std, n_limit=None):
        atk = GN(self, std=std)
        return get_acc(self, data_loader, n_limit=n_limit, atk=atk)
        
    def eval_rob_accuracy_fgsm(self, data_loader, eps, n_limit=None):
        atk = FGSM(self, eps=eps)        
        return get_acc(self, data_loader, n_limit=n_limit, atk=atk)
    
    def eval_rob_accuracy_pgd(self, data_loader, eps, alpha, steps, random_start=True, 
                              restart_num=1, norm='Linf', n_limit=None):
        if norm=='Linf':
            atk = PGD(self, eps=eps, alpha=alpha,
                      steps=steps, random_start=random_start)
        elif norm=='L2':
            atk = PGDL2(self, eps=eps, alpha=alpha,
                        steps=steps, random_start=random_start)
        else:
            raise ValueError('Invalid norm.')
            
        if restart_num > 1:
            atk = torchattacks.MultiAttack([atk]*restart_num)
        return get_acc(self, data_loader, n_limit=n_limit, atk=atk)
        
    def eval_rob_accuracy_autoattack(self, data_loader, eps, version='standard',
                                     norm='Linf', n_limit=None):
        atk = AutoAttack(self, norm=norm, eps=eps,
                         version=version, n_classes=self.n_classes)
        return get_acc(self, data_loader, n_limit=n_limit, atk=atk)
