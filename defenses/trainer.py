from .trainers.base_trainer import BaseTrainer
from .trainers.fgsm_adv_trainer import FGSMAdvTrainer
from .trainers.free_adv_trainer import FreeAdvTrainer
from .trainers.fast_adv_trainer import FastAdvTrainer
from .trainers.gradalign_adv_trainer import GradAlignAdvTrainer
from .trainers.pgd_adv_trainer import PGDAdvTrainer
from .trainers.trades_adv_trainer import TRADESAdvTrainer
from .trainers.mart_adv_trainer import MARTAdvTrainer

def get_trainer(name):
    if name == "Base":
        return BaseTrainer
        
    elif name == "FGSMAdv":
        return FGSMAdvTrainer
        
    elif name == "Free":
        return FreeAdvTrainer
        
    elif name == "GradAlign":
        return GradAlignAdvTrainer
        
    elif name == "PGDAdv":
        return PGDAdvTrainer
        
    elif name == "TRADES":
        return TRADESAdvTrainer
        
    elif name == "MART":
        return MARTAdvTrainer
        
    else:
        raise ValueError("Not valid trainer name.")