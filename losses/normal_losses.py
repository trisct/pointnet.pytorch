import torch
import torch.nn as nn
import torch.nn.functional as F
import math

DIV_EPSILON = 1e-8

def inner_prod_loss(input, target, accu_thresholds_in_deg=None):
    """
    input: [..., 3], where `...` for input and target are the same.
    target: [..., 3], where `...` for input and target are the same.
    accu_thresholds_in_deg: a list of thresholds in degrees.

    It computes the inner product (cos\theta) of the input prediction and the target normal,
    and uses 1-cos\theta as the loss.

    If accu_thresholds is not None, then the function returns a list of accuracies based on these thresholds
    """

    normalized_pred = input / ((input ** 2).sum(dim=-1, keepdim=True).sqrt() + DIV_EPSILON)
    normalized_target = target / ((target ** 2).sum(dim=-1, keepdim=True).sqrt() + DIV_EPSILON)

    inner_prod = (normalized_pred * normalized_target).sum(dim=-1)

    if accu_thresholds_in_deg is not None:
        accu_thresholds_in_cos = [ math.cos(threshold * math.pi / 180)\
            for threshold in accu_thresholds_in_deg ]
        accu_inner_prod = [ (inner_prod > threshold).sum() / inner_prod.view(-1).shape[0]\
            for threshold in accu_thresholds_in_cos]

    loss_inner_prod = 1 - inner_prod.mean()
    
    if accu_thresholds_in_deg is not None:
        return loss_inner_prod, accu_inner_prod
    return loss_inner_prod

def normalization_reg_loss(input):
    """
    input: [..., 3]

    It computes the length of each vector and uses the L2 loss between the lengths and 1.
    """

    lengths = (input ** 2).sum(dim=-1).sqrt()
    loss_norm_reg = ((lengths - 1) ** 2).mean()

    return loss_norm_reg