import torch
import torch.nn as nn

from ..advtrainer import AdvTrainer

class GradAlign(AdvTrainer):
    r"""
    GradAlign in 'Understanding and Improving Fast Adversarial Training'
    [https://arxiv.org/abs/2007.02617]
    [https://github.com/tml-epfl/understanding-fast-adv-training]

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
        alpha (float): alpha in the paper.
        grad_align_cos_lambda (float): parameter for the regularization term.
    """
    def __init__(self, model, eps, alpha, grad_align_cos_lambda):
        super().__init__("GradAlign", model)
        self.record_keys = ["Loss", "CALoss", "GALoss"] # Must be same as the items returned by self._do_iter
        self.eps = eps
        self.alpha = alpha
        self.grad_align_cos_lambda = grad_align_cos_lambda
    
    def _do_iter(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        X = images.to(self.device)
        Y = labels.to(self.device)

        # Calculate loss_gradalign
        X_new = torch.cat([X.clone(), X.clone()], dim=0)
        Y_new = torch.cat([Y.clone(), Y.clone()], dim=0)

        delta1 = torch.empty_like(X).uniform_(-self.eps, self.eps)
        delta2 = torch.empty_like(X).uniform_(-self.eps, self.eps)
        delta1.requires_grad = True
        delta2.requires_grad = True
        
        X_new[:len(X)] += delta1
        X_new[len(X):] += delta2
        X_new = torch.clamp(X_new, 0, 1)
        
        logits_new = self.model(X_new)
        loss_gn = nn.CrossEntropyLoss()(logits_new, Y_new)
        grad1, grad2 = torch.autograd.grad(loss_gn, [delta1, delta2],
                                           retain_graph=False, create_graph=False)
        
        X_adv = X_new[:len(X)] + self.alpha*grad1.sign()
        delta = torch.clamp(X_adv - X, min=-self.eps, max=self.eps).detach()
        X_adv = torch.clamp(X + delta, min=0, max=1).detach()
        
        grads_nnz_idx = ((grad1**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
        grad1, grad2 = grad1[grads_nnz_idx], grad2[grads_nnz_idx]
        grad1_norms = self._l2_norm_batch(grad1)
        grad2_norms = self._l2_norm_batch(grad2)
        grad1_normalized = grad1 / grad1_norms[:, None, None, None]
        grad2_normalized = grad2 / grad2_norms[:, None, None, None]
        cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))

        loss_gradalign = torch.tensor([0]).to(self.device)
        if len(cos) > 0:
            cos_aggr = cos.mean()
            loss_gradalign = (1.0 - cos_aggr)
        
        # Calculate loss_ce
        logits_adv = self.model(X_adv)
        loss_ce_adv = nn.CrossEntropyLoss()(logits_adv, Y)
        
        cost = loss_ce_adv + self.grad_align_cos_lambda * loss_gradalign
        
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost.item(), loss_ce_adv.item(), loss_gradalign.item()
    
    def _l2_norm_batch(self, v):
        norms = (v ** 2).sum([1, 2, 3]) ** 0.5
        # norms[norms == 0] = np.inf
        return norms
    
