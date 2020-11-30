import math

import torch


def nll_loss_manual(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    Manually computes the negative log likelihood of targets given preds where preds 
    consists of a mean and a std for each sample
    '''

    mean = pred[0]
    sigma = pred[1]
    NLL = torch.mean(torch.log(sigma) + 0.5 * torch.pow((target - mean) / sigma, 2))
    return NLL


def nll_loss(dist, target: torch.Tensor):
    '''
    Compoute the negative log likelihood of targets given the guassian distributions dist.
    '''
    # we must return a scalar as that what pytorch requires for backpropagation
    return -dist.log_prob(target).sum()
