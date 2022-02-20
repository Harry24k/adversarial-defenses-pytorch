import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import copy

from torchattacks import TPGD

from ..advtrainer import AdvTrainer

EPS = 1E-20

def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict

def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class TradesAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, inputs_clean, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss_natural = F.cross_entropy(self.proxy(inputs_clean), targets)
        loss_robust = F.kl_div(F.log_softmax(self.proxy(inputs_adv), dim=1),
                               F.softmax(self.proxy(inputs_clean), dim=1),
                               reduction='batchmean')
        loss = - 1.0 * (loss_natural + beta * loss_robust)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


class AwpTRADES(AdvTrainer):
    r"""
    Adversarial training with TRADES in 'Adversarial Weight Perturbation Helps Robust Generalization'
    [https://arxiv.org/pdf/2004.05884]
    [https://github.com/csdongxian/AWP]

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
        awp_gamma (float): regularization parameter for AWP.
        proxy_lr (float): learning rate of proxy model in AWP.

    """
    def __init__(self, model, eps, alpha, steps, beta, awp_gamma=0.01, proxy_lr=0.01):
        super().__init__("AwpTRADES", model)
        self.record_keys = ["Loss", "CELoss", "KLLoss"] # Must be same as the items returned by self._do_iter
        self.atk = TPGD(model, eps, alpha, steps)
        self.beta = beta

        self.proxy = copy.deepcopy(self.model.model)
        self.proxy_opt = torch.optim.SGD(self.proxy.parameters(), lr=proxy_lr)
        self.awp_adversary = TradesAWP(model=self.model.model, proxy=self.proxy, proxy_optim=self.proxy_opt, gamma=awp_gamma)

    def _do_iter(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        X = images.to(self.device)
        Y = labels.to(self.device)

        X_adv = self.atk(X)

        awp = self.awp_adversary.calc_awp(inputs_adv=X_adv,inputs_clean=X,targets=Y,beta=self.beta)
        self.awp_adversary.perturb(awp)

        logits_clean = self.model(X)
        loss_ce = nn.CrossEntropyLoss()(logits_clean, Y)

        logits_adv = self.model(X_adv)
        prob_clean = F.softmax(logits_clean, dim=1)
        log_prob_adv = F.log_softmax(logits_adv, dim=1)
        loss_kl = nn.KLDivLoss(reduction='batchmean')(log_prob_adv, prob_clean)

        cost = loss_ce + self.beta * loss_kl

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        self.awp_adversary.restore(awp)

        return cost.item(), loss_ce.item(), loss_kl.item()
