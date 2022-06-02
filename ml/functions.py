from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
import os
import torch
from torch.nn import functional as F
from torch import optim
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss
import numpy as np

from contextlib import contextmanager

logger = logging.getLogger(__name__)

def get_activation(activation):
    if activation == "relu":
        return torch.relu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return torch.sigmoid
    elif activation == "log_sigmoid":
        return F.logsigmoid
    else:
        raise ValueError("Activation function %s unknown", activation)


def get_loss(method, alpha, w = 1, loss_type="regular"):
    if method in ["carl", "carl2"]:
        loss_functions = [ratio_xe(loss_type)]
        # It is advised not to use loss_weights inside this function like this
        loss_weights = [1.0] #sjiggins
        loss_labels = ["xe"]
    else:
        raise NotImplementedError("Unknown method {}".format(method))

    return loss_functions, loss_labels, loss_weights #sjiggins


def get_optimizer(optimizer, nesterov_momentum):
    opt_kwargs = None
    if optimizer == "adam":
        opt = optim.Adam
    elif optimizer == "amsgrad":
        opt = optim.Adam
        opt_kwargs = {"amsgrad": True}
    elif optimizer == "sgd":
        opt = optim.SGD
        if nesterov_momentum is not None:
            opt_kwargs = {"momentum": nesterov_momentum}
    else:
        raise ValueError("Unknown optimizer {}".format(optimizer))
    return opt, opt_kwargs


def _ratio_xe(s_hat, y_true, w):
    if w is None:
        w = torch.ones(y_true.shape[0])
    loss = BCELoss(weight=w, reduction='mean')(s_hat, y_true)

    return loss

def _ratio_xe_prob_reg(s_hat, y_true, w):
    if w is None:
        w = torch.ones(y_true.shape[0])
    # Calculate BCE loss
    bceloss = BCELoss(weight=w, reduction='none')
    loss = bceloss(s_hat, y_true)

    # Calculate the suppresion term - This is all static at present must change to allow user hyperparameter optimisation
    s_hat_temp = torch.sub(s_hat, 0.5)
    s_hat_temp = torch.where(s_hat_temp > 0, s_hat_temp, torch.zeros(s_hat_temp.size()).to("cuda", torch.float, non_blocking=True))
    s_hat_temp = torch.mul(s_hat_temp, 2)
    s_hat_temp = torch.pow(s_hat_temp, 4)
    inverse_sub = torch.reciprocal(1-s_hat_temp)

    # Calculate the final suppression term per event weighted but the coefficient
    coefficient=0.1
    inverse_sub = torch.mul(inverse_sub, coefficient)
    # Add the suppression term
    loss = torch.add(loss, inverse_sub)
    # Need to return scalar
    loss = torch.sum(loss)

    return loss


def _ratio_xe_abs_w(s_hat, y_true, w):
    if w is None:
        w = torch.ones(y_true.shape[0])
    loss = BCELoss(weight=torch.abs(w))(s_hat, y_true)
    return loss

def _ratio_xe_log_abs_w(s_hat, y_true, w):
    if w is None:
        w = torch.ones(y_true.shape[0])
    loss = BCELoss(weight=torh.log(torch.abs(w)))(s_hat, y_true)
    return loss

def ratio_xe(type):
    """
    server as factory for different weight modification.
    """
    preserved_type = {
        "regular": _ratio_xe,
        "score_suppressed": _ratio_xe_prob_reg,
        "abs(w)": _ratio_xe_abs_w,
        "log(abs(w))": _ratio_xe_log_abs_w,
    }
    try:
        return preserved_type[type]
    except KeyError as _error:
        raise KeyError(f"ratio_xe requires one of the type from: {preserved_type}") from _error


@contextmanager
def less_logging():
    """
    Silences INFO logging messages. Based on https://gist.github.com/simon-weber/7853144
    """

    if logging.root.manager.disable != logging.DEBUG:
        yield
        return

    try:
        logging.disable(logging.INFO)
        yield
    finally:
        logging.disable(logging.DEBUG)
