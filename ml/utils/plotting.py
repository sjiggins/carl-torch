from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import logging
import root_numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPRegressor


import torch
from .tools import create_missing_folders

logger = logging.getLogger(__name__)

def draw_unweighted_distributions(x0, x1, weights, varis, vlabels, binning, legend):
    hist_settings0 = {'alpha': 0.3}
    hist_settings1 = {'histtype':'step', 'color':'black', 'linewidth':1, 'linestyle':'--'}
    columns = range(len(varis))
    for id, column in enumerate(columns, 1):
        plt.figure(figsize=(5, 4))
        plt.yscale('log')
        plt.hist(x0[:,column], bins = binning[id-1],weights=weights, label = legend[0], **hist_settings0)
        plt.hist(x1[:,column], bins = binning[id-1],label = legend[1], **hist_settings1)
        plt.xlabel('%s'%(vlabels[id-1])) 
        plt.legend(frameon=False)
        axes = plt.gca()
        axes.set_ylim([1,10000000])                  
        create_missing_folders(["plots"])                                                              
        plt.savefig("plots/unweighted_%s_%sVs%s.png"%(varis[id-1], legend[0],legend[1]))                                                                                                        
        plt.clf()

def draw_weighted_distributions(x0, x1, weights, varis, vlabels, binning, label, legend):
    hist_settings0 = {'alpha': 0.3}
    hist_settings1 = {'histtype':'step', 'color':'black', 'linewidth':1, 'linestyle':'--'}
    columns = range(len(varis))
    for id, column in enumerate(columns, 1):
        plt.figure(figsize=(5, 4))
        plt.yscale('log')
        plt.hist(x0[:,column], bins = binning[id-1], label = legend[0], **hist_settings0)
        plt.hist(x0[:,column], bins = binning[id-1],weights=weights, label = legend[0]+'*CARL', **hist_settings0)
        plt.hist(x1[:,column], bins = binning[id-1],label = legend[1], **hist_settings1)
        plt.xlabel('%s'%(vlabels[id-1])) 
        plt.legend(frameon=False,title = '%s sample'%(label) )
        axes = plt.gca()
        axes.set_ylim([1,10000000])                  
        create_missing_folders(["plots"])                                                              
        plt.savefig("plots/weighted_%s_%sVs%s_%s.png"%(varis[id-1], legend[0],legend[1],label))                                                                                                        
        plt.clf()
def weight_data(x0,x1,weights, max_weight=10000.):
    x1_len = x1.shape[0]
    x0_len = x0.shape[0]
    weights[weights>max_weight]=max_weight

    weights = weights / weights.sum()
    weighted_data = np.random.choice(range(x0_len), x0_len, p = weights)
    w_x0 = x0.copy()[weighted_data]
    y = np.zeros(x1_len + x0_len)
    x_all = np.vstack((w_x0,x1))
    y_all = np.zeros(x1_len +x0_len)
    y_all[x0_len:] = 1
    return (x_all,y_all)

def resampled_discriminator_and_roc(original, target, weights):
    (data, labels) = weight_data(original,target,weights)
    W = np.concatenate([weights / weights.sum() * len(target), [1] * len(target)])

    Xtr, Xts, Ytr, Yts, Wtr, Wts = train_test_split(data, labels, W, random_state=42, train_size=0.51, test_size=0.49)    
    
    discriminator = MLPRegressor(tol=1e-05, activation="logistic", 
               hidden_layer_sizes=(10, 10), learning_rate_init=1e-07, 
               learning_rate="constant", solver="lbfgs", random_state=1, 
               max_iter=75)

    discriminator.fit(Xtr,Ytr)
    predicted = discriminator.predict(Xts)
    fpr, tpr, _  = roc_curve(Yts,predicted.ravel())
    roc_auc = auc(fpr, tpr)
    return fpr,tpr,roc_auc
 
def draw_ROC(X0, X1, weights, label, legend):
    no_weights_scaled = np.ones(X0.shape[0])/np.ones(X0.shape[0]).sum() * len(X1)
    fpr_t,tpr_t,roc_auc_t = resampled_discriminator_and_roc(X0, X1, no_weights_scaled)
    plt.plot(fpr_t, tpr_t, label=r"no weight, AUC=%.3f" % roc_auc_t)
    fpr_tC,tpr_tC,roc_auc_tC = resampled_discriminator_and_roc(X0, X1, weights)
    plt.plot(fpr_tC, tpr_tC, label=r"CARL weight, AUC=%.3f" % roc_auc_tC)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0]) 
    plt.ylim([0.0, 1.05])
    plt.title('Resampled proportional to weights')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", title = label)
    plt.tight_layout()
    plt.savefig('plots/roc_resampled_%sVs%s_%s.png'%(legend[0], legend[1],label)) 
    print("CARL weighted %s AUC is %.3f"%(label,roc_auc_tC))
    print("Unweighted %s AUC is %.3f"%(label,roc_auc_t))