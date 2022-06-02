from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
# from torch.nn import functional as F
# from torch.autograd import grad
from .functions import get_activation

import logging

logger = logging.getLogger(__name__)

class RatioModel(nn.Module):

    def __init__(self, n_observables, n_hidden, activation="relu", dropout_prob=0.1):

        super(RatioModel, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation(activation)
        self.dropout_prob = dropout_prob

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables
        
        # Hidden layers
        logger.info("Building {} hidden layers building".format(n_hidden))
        for n_hidden_units in n_hidden:
            if self.dropout_prob > 1.0e-9:
                logger.info("Layer {} will contain dropout nodes with p={}".format(n_hidden_units,self.dropout_prob))
                self.layers.append(nn.Dropout(self.dropout_prob))
            self.layers.append(nn.Linear(n_last, n_hidden_units))
            n_last = n_hidden_units

        # Log r layer
        if self.dropout_prob > 1.0e-9:
            self.layers.append(nn.Dropout(self.dropout_prob))
        self.layers.append(nn.Linear(n_last, 1))

    def forward(self, x: torch.Tensor):
        s_hat = x
        for i, layer in enumerate(self.layers):
            if i > 0:
                s_hat = self.activation(s_hat)
            s_hat = layer(s_hat)
        s_hat = torch.sigmoid(s_hat)
        # clamping very small value to 1e-9 to avoid zero
        s_hat = torch.clamp(s_hat, min=1.0e-9)
        r_hat = (1 - s_hat) / s_hat

        return r_hat, s_hat

    def to(self, *args, **kwargs):
        self = super(RatioModel, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self
