from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import math
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPRegressor
from sklearn.calibration import calibration_curve


import torch
from .tools import create_missing_folders

logger = logging.getLogger(__name__)
hist_settings_nom = {'alpha': 0.25, 'color':'blue'}
hist_settings_alt = {'alpha': 0.25, 'color':'orange'}
hist_settings_CARL = {'histtype':'step', 'color':'black', 'linewidth':1, 'linestyle':'--'}
hist_settings_CARL_ratio = {'color':'black', 'linewidth':1, 'linestyle':'--'}
#hist_settings1_step = {'color':'black', 'linewidth':1, 'linestyle':'--'}


def draw_unweighted_distributions(x0, x1,
                                  weights,
                                  variables,
                                  vlabels,
                                  binning,
                                  legend,
                                  n,
                                  save = False):
    plt.figure(figsize=(14, 10))
    columns = range(len(variables))
    for id, column in enumerate(columns, 1):
        if save: plt.figure(figsize=(5, 4.2))
        else: plt.subplot(3,4, id)
        plt.yscale('log')
        plt.hist(x0[:,column], bins = binning[id-1], weights=weights, label = "nominal", **hist_settings_nom)
        plt.hist(x1[:,column], bins = binning[id-1], label = legend, **hist_settings_alt)
        plt.xlabel('%s'%(vlabels[id-1]), horizontalalignment='right',x=1)
        plt.legend(frameon=False)
        axes = plt.gca()
        axes.set_ylim([len(x0)*0.001,len(x0)*2])
        if save:
            create_missing_folders(["plots"])
            plt.savefig("plots/%s_nominalVs%s_%s.png"%(variables[id-1],legend, n))
            plt.clf()
            plt.close()

def draw_weighted_distributions(x0, x1, w0, w1,
                                weights,
                                variables,
                                binning, label,
                                legend,
                                n,
                                save = False):
    plt.figure(figsize=(14, 10))
    #columns = range(len(variables))

    for id, column in enumerate(variables):
        print("<plotting.py::draw_weighted_distribution()>::   id: {},   column: {}".format(id,column))
        print("<plotting.py::draw_weighted_distribution()>::     binning: {}".format(binning[id]))
        if save: plt.figure(figsize=(5, 4))
        else: plt.subplot(3,4, id)
        #plt.yscale('log')
        #plt.hist(x0[:,id], bins = binning[column], label = "nominal", **hist_settings0)
        #plt.hist(x0[:,id], bins = binning[column], weights=weights, label = 'nominal*CARL', **hist_settings0)
        #plt.hist(x1[:,id], bins = binning[column], label = legend, **hist_settings1)
        w0 = w0.flatten()
        w1 = w1.flatten()
        w_carl = w0*weights
        plt.hist(x0[:,id], bins = binning[id], weights = w0, label = "nominal", **hist_settings_nom)
        plt.hist(x0[:,id], bins = binning[id], weights = w_carl, label = 'nominal*CARL', **hist_settings_CARL)
        plt.hist(x1[:,id], bins = binning[id], weights = w1, label = "alternative", **hist_settings_alt)
        plt.xlabel('%s'%(column), horizontalalignment='right',x=1)
        plt.legend(frameon=False,title = '%s sample %s'%(label,legend) )
        axes = plt.gca()
        #axes.set_ylim([len(x0)*0.001,len(x0)*2]) #sjiggins
        #axes.set_ylim([w0.sum()*0.001,w0.sum()*2]) #sjiggins
        if save:

            # Create folder for storing plots
            create_missing_folders([f"plots/{legend}"])
            # Form output name and then save
            output_name = f"plots/{legend}/w_{column}_nominalVs{legend}_{label}_{n}"
            plt.savefig(f"{output_name}.png")
            plt.clf()
            plt.close()

            # ratio plot
            x0_hist, edge0 = np.histogram(x0[:,id], bins = binning[id], weights = w0)
            x1_hist, edge1 = np.histogram(x1[:,id], bins = binning[id], weights = w1)
            carl_hist, edgecarl = np.histogram(x0[:,id], bins = binning[id], weights = w_carl)
            #x1_ratio = x1_hist/x0_hist
            x1_ratio = x0_hist/x1_hist
            #carl_ratio = carl_hist/x0_hist
            carl_ratio = carl_hist/x1_hist
            # Generate reference line
            #   -> Extract the lowest and highest bin edge
            xref= [binning[id].min(), binning[id].max()]
            #   -> Now generate the x and y points of the reference line
            yref = [1.0,1.0]

            ## Generate error bands for the reference histogram
            x0_error = []
            x1_error = []
            if len(binning[id]) > 1:
                width = abs(binning[id][1] - binning[id][0] )
                for xbin in binning[id]:
                    # Form masks for all event that match condition
                    mask0 = (x0[:,id] < (xbin + width)) & (x0[:,id] > (xbin - width))
                    mask1 = (x1[:,id] < (xbin + width)) & (x1[:,id] > (xbin - width))
                    # Form bin error
                    binsqrsum_x0 = np.sum(w0[mask0]**2)
                    binsqrsum_x1 = np.sum(w1[mask1]**2)
                    binsqrsum_x0 = math.sqrt(binsqrsum_x0)
                    binsqrsum_x1 = math.sqrt(binsqrsum_x1)
                    # Form relative error
                    binsqrsum_x0 = binsqrsum_x0/w0[mask0].sum()
                    binsqrsum_x1 = binsqrsum_x1/w1[mask1].sum()

                    x0_error.append(binsqrsum_x0 if binsqrsum_x0 > 0 else 0.0 )
                    x1_error.append(binsqrsum_x1 if binsqrsum_x1 > 0 else 0.0)
                    #print("binsqrsum_x0:  {}".format(binsqrsum_x0))
                    #print("binsqrsum_x1:  {}".format(binsqrsum_x1))


            #else:
            #    # Fill in errors with 0 as could not determine bin width
            #    x0_hist_error =
            #    x1_hist_error =


            # Convert error lists to numpy arrays
            x0_error = np.array(x0_error)
            x1_error = np.array(x1_error)

            plt.step( xref, yref, where="post", label=legend+" / "+legend, **hist_settings_alt )
            plt.step( edge1[:-1], x1_ratio, where="post", label="nom / "+legend, **hist_settings_nom)
            plt.step( edgecarl[:-1], carl_ratio, where="post", label = '(nominal*CARL) / '+legend, **hist_settings_CARL_ratio)
            #plt.fill_between(edge[:,-1], 1.0, x1_ratio)
            yref_error = np.ones(len(edge1))
            yref_error_up = 2* np.sqrt( np.power(x1_error,2) + np.power(x0_error, 2)) # height from bottom
            yref_error_down = yref_error - np.sqrt(np.power(x1_error, 2) + np.power(x0_error,2))
            print("edg1e:    {}".format(edge1))
            print("yref_error:    {}".format(yref_error))
            print("x0_error:    {}".format(x0_error))
            print("x1_error:    {}".format(x1_error))
            print("width:       {}".format(np.diff(edge1)))
            #plt.bar( x=edge1, height=yref_error+x1_error, bottom = yref_error-x1_error, width=np.diff(edge1), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1, label='uncertainty band')
            plt.bar( x=edge1[:-1],
                     height=yref_error_up[:-1], bottom = yref_error_down[:-1],
                     color='red',
                     width=np.diff(edge1),
                     align='edge',
                     alpha=0.25,
                     label='uncertainty band')
            #         #width=np.diff(edge1),
            #         #align='edge',
            #         #linewidth=0,
            #         #color='red',
            #         #alpha=0.25,
            #         #label='uncertainty band')

            plt.xlabel('%s'%(column), horizontalalignment='right',x=1)
            plt.legend(frameon=False,title = '%s sample'%(label) )
            axes = plt.gca()
            axes.set_ylim([0.5, 1.6])
            plt.yticks(np.arange(0.5,1.6,0.1))
            plt.savefig(f"{output_name}_ratio.png")
            plt.clf()
            plt.close()

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

def draw_ROC(X0, X1, W0, W1, weights, label, legend, n, plot = True):
    plt.figure(figsize=(4, 3))
    no_weights_scaled = np.ones(X0.shape[0])/np.ones(X0.shape[0]).sum() * len(X1) #sjiggins
    print("no_weights_scaled:  {}".format(no_weights_scaled))
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
    if plot:
        plt.savefig('plots/roc_nominalVs%s_%s_%s.png'%(legend,label, n))
        plt.clf()
    logger.info("CARL weighted %s AUC is %.3f"%(label,roc_auc_tC))
    logger.info("Unweighted %s AUC is %.3f"%(label,roc_auc_t))
    logger.info("Saving ROC plots to /plots")

def plot_calibration_curve(y, probs_raw, probs_cal, do, var, save = False):
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    frac_of_pos_raw, mean_pred_value_raw = calibration_curve(y, probs_raw, n_bins=50)
    frac_of_pos_cal, mean_pred_value_cal = calibration_curve(y, probs_cal, n_bins=50)

    ax1.plot(mean_pred_value_raw, frac_of_pos_raw, "s-", label='uncalibrated', **hist_settings0)
    ax1.plot(mean_pred_value_cal, frac_of_pos_cal, "s-", label='calibrated', **hist_settings0)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration plot')

    ax2.hist(probs_raw, range=(0, 1), bins=50, label='uncalibrated', lw=2, **hist_settings0)
    ax2.hist(probs_cal, range=(0, 1), bins=50, label='calibrated', lw=2, **hist_settings0)
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    if save:
        plt.savefig('plots/calibration_'+do+'_'+var+'.png')
        plt.clf()
    logger.info("Saving calibration curves to /plots")

def draw_weights(weightCT, weightCA, legend, do, n, save = False):
    plt.yscale('log')
    plt.hist(weightCT, bins = np.exp(np.linspace(-0.5,1.1,50)), label = 'carl-torch', **hist_settings0)
    plt.hist(weightCA, bins = np.exp(np.linspace(-0.5,1.1,50)), label = 'carlAthena', **hist_settings0)
    plt.xlabel('weights', horizontalalignment='right',x=1)
    plt.legend(frameon=False)
    plt.savefig("plots/weights_%s_%s_%s.png"%(do, legend, n))
    plt.clf()
    plt.close()

def draw_scatter(weightsCT, weightsCA, legend, do, n):
    print("weights carl-torch ", len(weightsCT))
    print("weights carlAthena ", len(weightsCA))
    plt.scatter(weightsCT, weightsCA, alpha=0.5)
    max_temp=1.5
    plt.plot([0,max_temp],[0,max_temp], lw=2, c='r')
    plt.xlim(0,max_temp)
    plt.ylim(0,max_temp)
    plt.xlabel('weights carl-torch')
    plt.ylabel('weights carlAthena')
    plt.savefig("plots/scatter_weights_%s_%s_%s.png"%(do, legend, n))
    plt.clf()
    plt.close()
