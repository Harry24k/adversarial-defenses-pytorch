import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks import PGD, TPGD

from ..advtrainer import AdvTrainer

class ATRADES(AdvTrainer):
    r"""
    AT + TRADES

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
        alpha (float): step size.
        steps (int): number of steps.
        beta (float): trade-off regularization parameter.
        eta (float): the ratio between the loss of AT and TRADES. (eta=1 is equal to AT)
        inner_loss (str): inner loss to maximize ['ce', 'kl']. 
    """
    def __init__(self, model, eps, alpha, steps, beta, eta, inner_loss='ce'):
        super().__init__("TRADES", model)
        self.record_keys = ["Loss", "CELoss", "KLLoss"] # Must be same as the items returned by self._do_iter
        if inner_loss == 'ce':
            self.atk = PGD(model, eps, alpha, steps)
        elif inner_loss == 'kl':
            self.atk = TPGD(model, eps, alpha, steps)  
        else:
            raise ValueError("Not valid inner loss.")
        self.beta = beta
        self.eta = eta
        
    def _do_iter(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        X = images.to(self.device)
        Y = labels.to(self.device)

        X_adv = self.atk(X, Y)
        
        # AT
        cost_at = nn.CrossEntropyLoss()(logits_adv, Y)

        # TRADES
        logits_clean = self.model(X)
        loss_ce = nn.CrossEntropyLoss()(logits_clean, Y)
        
        logits_adv = self.model(X_adv)
        prob_clean = F.softmax(logits_clean, dim=1)
        log_prob_adv = F.log_softmax(logits_adv, dim=1)
        loss_kl = nn.KLDivLoss(reduction='batchmean')(log_prob_adv, prob_clean)
        
        cost_trades = loss_ce + self.beta * loss_kl
        
        # Combine        
        cost = self.eta*cost_at + (1-self.eta)*cost_trades
        
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost.item(), cost_at.item(), cost_trades.item()
