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


class EnsembleRatioModel(nn.Module):

    def __init__(self, 
                 n_observables, n_hidden, pairings,
                 n_subnetworks, n_subnetwork_hidden, subnetwork_activation,
                 activation="relu",   # May want to allow variable activations for each subnetwork
                 dropout_prob=0.1 ): 

        super(EnsembleRatioModel, self).__init__()

        # Save input
        self.n_hidden = n_hidden  # int()
        self.activation = get_activation(activation) # str()
        self.dropout_prob = dropout_prob  # float()
        self.pairings = pairings # list of tuples [(), ()]
        
        # Sub-Network architecture information stored as list of lists
        #   n_subnetwork_hidden = list[ [], [] ]
        self.n_subnetwork = n_subnetwork

        # Build the main MLP network
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

        # latent space layer dimensionality
        #   -> For now it uses the last layer of the hidden layers but no dropout
        #if self.dropout_prob > 1.0e-9:
        #    self.layers.append(nn.Dropout(self.dropout_prob))
        

        # Build the sub-networks
        #   subnetwork_id = int()
        #   subnetwork_layers = list[]
        for subnetwork_id,subnetwork_layers in enumerate(self.n_subnetwork_hidden):
            self.subnetworks = RatioModel( self.n_hidden[-1], subnetwork_layers, # Last hidden layer of the main MLP will be the latent space 
                                           self.activation, sel.dropout_prob ) 


    def forward(self, x: torch.Tensor, y : torch.tensor):

        # x:   input features from the data
        # y:   labels of the data

        # Latent Space MLP passing 
        repr = x
        for i, layer in enumerate(self.layers):
            if i > 0:
                repr = self.activation(repr)
            repr = layer(repr)
        
        # Split latent space representation 'repr' into the sub-networks using label masking
        #    label_id  = int() from enumeration which should align with labels of data
        #                For the positive intended use:
        #                       c+   =   positive weighted events of class y = 0,  y' = 0
        #                       c-   =   negative weighted events of class y = 1,  y' = 1
        #                       c'+  =   positive weighted events of class y = 0,  y' = 2
        #                       c'-  =   negative weighted events of class y = 1,  y' = 3
        label_masks = [] #torch.empty() #torch.empty( y.size(), dtype=torch.long )
        labelled_y = []
        labelled_x = []
        for label_id in torch.unique( y, dtype=torch.long):
            label_masks.append( y[y==label_id] )   # define the y-label masks
            masked_y.append( y[label_masks[-1]] ) # define the masked y-labels
            masked_repr.append( repr[ np.array(label_masks[-1]) ] )  # define the masked data

        # Loop through the pairings of the labelled data, composing the concatenated data
        #   -> The number of pairings and number of sub-networks should match. 
        #   -> The order of the pairs defines the sub-network identifier ordering and thus the sub-network is 
        #      optimised to classify said ordering
        probs = []
        labels = []
        for pair_id,pair in enumerate(self.pairings):
            
            # Filter the latent representation based on the pairing label_masks
            masked_y = torch.cat( (masked_y[pair[0]], masked_y[pair[-1]]), dim=0)
            masked_repr = torch.cat( (masked_repr[pair[0]], masked_repr[pair[-1]]), dim=0)

            # Probability of class labels from RatioModel instances using
            # categorical classification
            #   probs = list[ torch.tensor ]
            probs.append( self.subnetworks[pair_id](masked_repr) )
            
            # Store the labels for the loss calculation
            labels.append(masked_y)
            
        return repr, probs, labels

    def to(self, *args, **kwargs):
        self = super(EnsembleRatioModel, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self
