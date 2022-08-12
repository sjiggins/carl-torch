from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import logging
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import multiprocessing
import math
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.calibration import calibration_curve
import wasserstein
import scipy as scipy

import torch
from .tools import create_missing_folders

logger = logging.getLogger(__name__)
hist_settings_nom = {'alpha': 0.25, 'color':'blue'}
hist_settings_alt = {'alpha': 0.25, 'color':'orange'}
hist_settings_CARL = {'histtype':'step', 'color':'black', 'linewidth':1, 'linestyle':'--'}
hist_settings_CARL_ratio = {'color':'black', 'linewidth':1, 'linestyle':'--'}
#hist_settings1_step = {'color':'black', 'linewidth':1, 'linestyle':'--'}

do_dbug_plot = False
if do_dbug_plot:
    try:
        import pickle
        with open("addInvSample.pkl", "rb") as f:
            addInvSample = pickle.load(f)
    except Exception as e:
        print(e)
        addInvSample = None
else:
    addInvSample = None


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
        axes.set_ylim([len(x0)*0.001,len(x0)*3])
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
                                save = False,
                                ext_plot_path=None,
                                normalise=True):
    # Formatting
    font = font_manager.FontProperties(family='Symbol',
                                       style='normal', size=12)
    plt.rcParams['legend.title_fontsize'] = 14


    for id, column in enumerate(variables):
        #fig, axes = plt.subplots(3, sharex=True, figsize=(12,10))
        fig = plt.figure(figsize=(12,10))
        gs = fig.add_gridspec(3, hspace=0, height_ratios=[5,2,2])
        axes = gs.subplots(sharex=True)
        fig.suptitle("Differential Cross-section & Mapping Performance")
        print("<plotting.py::draw_weighted_distribution()>::   id: {},   column: {}".format(id,column))
        print("<plotting.py::draw_weighted_distribution()>::     binning: {}".format(binning[id]))
        #if save: axes[0].figure(figsize=(14, 10))
        #else: axes[0].plt.subplot(3,4, id)
        #plt.yscale('log')
        w0 = w0.flatten()
        w1 = w1.flatten()
        w_carl = w0*weights
        if normalise:
            w0 = w0/w0.sum()
            w1 = w1/w1.sum()
            w_carl = w_carl/w_carl.sum()
        nom_count, nom_bin, nom_bars = axes[0].hist(x0[:,id], bins = binning[id], weights = w0, label = "nominal", **hist_settings_nom, density=True)
        carl_count, carl_bin, carl_bars = axes[0].hist(x0[:,id], bins = binning[id], weights = w_carl, label = 'nominal*CARL', **hist_settings_CARL, density=True)
        alt_count, alt_bins, alt_bars = axes[0].hist(x1[:,id], bins = binning[id], weights = w1, label = legend, **hist_settings_alt, density=True)
        axes[0].grid(axis='x', color='silver')
        if addInvSample:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            _setting = {'histtype':'step', 'linewidth':2, 'color':'red'}
            _x0 = addInvSample[0].to_numpy()
            _w0 = addInvSample[1].to_numpy().flatten()
            axes[0].hist(_x0[:,id], bins = binning[id], weights = _w0, label = "Non-splited nominal Inverted fraction", **_setting)

            _inv_setting = {'histtype':'step', 'linewidth':3, 'color':'pink'}
            if column == "polarity":
                mul_scale = -1
            else:
                mul_scale = 1
            _w0= (addInvSample[1]*-1).to_numpy().flatten()
            axes[0].hist(_x0[:,id]*mul_scale, bins = binning[id], weights = _w0, label = "Non-splited nominal fraction", **_inv_setting)

        axes[0].set_xlabel('%s'%(column), horizontalalignment='right',x=1)
        axes[0].legend(frameon=False,title = '%s sample'%(label), prop=font )
        axes[0].set_ylabel(r"$\frac{1}{N} \cdot \frac{d \sigma}{dx}$", horizontalalignment='center',x=1, fontsize=18)
        #axes[0].legend.get_title().set_fontsize('14')
        #axes = plt.gca()
        #axes[0].set_xticks(fontsize=14)
        #axes[0].set_yticks(fontsize=14)

        # Calculate the chi^[2}
        nom_alt_chi, nom_alt_p = scipy.stats.chisquare(f_obs=nom_count, f_exp=alt_count)
        carl_alt_chi, carl_alt_p = scipy.stats.chisquare(f_obs=carl_count, f_exp=alt_count)
        # Result string
        nom_alt_chi_res = "$\chi^{2}$ ="+" {},  p-value = {}".format(nom_alt_chi,nom_alt_p)
        carl_alt_chi_res = "$\chi^{2}$ ="+" {},  p-value = {}".format(carl_alt_chi,carl_alt_p)
        logger.info("{}".format(nom_alt_chi_res))
        logger.info("{}".format(carl_alt_chi_res))
        
        # KL-divergence
        ########### SciPy ##########
        #nom_alt_KL = scipy.special.kl_div(nom_count, alt_count)
        #carl_alt_KL = scipy.special.kl_div(carl_count, alt_count)
        ## Result string
        #nom_alt_KL_res = "KL(nom,alt) = {}".format(nom_alt_KL)
        #carl_alt_KL_res = "KL(carl,alt) = {}".format(carl_alt_KL)
        ############################
        ########### Custom #########
        nom_alt_KL = compute_kl_divergence(x0[:,id], w0, x1[:,id], w1, len(binning[id]))
        #carl_alt_KL = compute_kl_divergence(x0[:,id], w0, x0[:,id], w_carl, len(binning[id]))
        carl_alt_KL = compute_kl_divergence(x0[:,id], w_carl, x1[:,id], w1, len(binning[id]))
        ## Result string
        nom_alt_KL_res = "KL(nom,alt) = {}".format(nom_alt_KL)
        carl_alt_KL_res = "KL(carl,alt) = {}".format(carl_alt_KL)
        ############################        
        logger.info("{}".format(nom_alt_KL_res))
        logger.info("{}".format(carl_alt_KL_res))

        # Calculate the EMD
        #emd = wasserstein.EMD()
        #emd_var_val = emd(w0, x0[:,id], w1, x1[:,id])
        #print('EMD:', emd)
        #print('EMD var. value:', emd_var_val)

        # Ad metrics to plot
        #axes[0].text(0.2, 0.9, s=nom_alt_chi_res, horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
        #axes[0].text(0.2, 0.85, s=carl_alt_chi_res, horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
        #axes[0].text(0.2, 0.8, s=nom_alt_KL_res, horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
        #axes[0].text(0.2, 0.75, s=carl_alt_KL_res, horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

        # Calculate maximum and minimum of y-axis
        y_min, y_max = axes[0].get_ylim()
        axes[0].set_ylim([y_min*0.9, y_max*1.5])
        if save:
            # Create folder for storing plots and Form output name and then save
            if ext_plot_path:
                create_missing_folders([f"plots/{legend}/{ext_plot_path}"])
                output_name = f"plots/{legend}/{ext_plot_path}/w_{column}_nominalVs{legend}_{label}_{n}"
            else:
                create_missing_folders([f"plots/{legend}"])
                output_name = f"plots/{legend}/w_{column}_nominalVs{legend}_{label}_{n}"

            #plt.savefig(f"{output_name}.png")
            #plt.clf()
            #plt.close()
            #plt.figure(figsize=(10, 8)) # this line is needed to keep same canvas size

            # ratio plot
            x0_hist, edge0 = np.histogram(x0[:,id], bins = binning[id], weights = w0, density=True)
            x1_hist, edge1 = np.histogram(x1[:,id], bins = binning[id], weights = w1, density=True)
            carl_hist, edgecarl = np.histogram(x0[:,id], bins = binning[id], weights = w_carl, density=True)
            #x1_ratio = x1_hist/x0_hist
            x1_ratio = x0_hist/x1_hist
            #carl_ratio = carl_hist/x0_hist
            carl_ratio = carl_hist/x1_hist
            # Generate reference line
            #   -> Extract the lowest and highest bin edge
            xref= [binning[id].min(), binning[id].max()]
            #   -> Now generate the x and y points of the reference line
            yref = [1.0,1.0]

            ## Generate error bands and residue for the reference histogram
            x0_error = []
            x1_error = []
            residue = []
            residue_carl = []
            # Normalise weights to unity
            w0 = w0*(1.0/np.sum(w0))
            w1 = w1*(1.0/np.sum(w1))
            w_carl = w_carl*(1.0/np.sum(w_carl))
            if len(binning[id]) > 1:
                width = abs(binning[id][1] - binning[id][0] )
                for xbin in binning[id]:
                    # Form masks for all event that match condition
                    mask0 = (x0[:,id] < (xbin + width)) & (x0[:,id] > (xbin - width))
                    mask1 = (x1[:,id] < (xbin + width)) & (x1[:,id] > (xbin - width))
                    # Form bin error
                    binsqrsum_x0 = np.sum(w0[mask0]**2)
                    binsqrsum_x1 = np.sum(w1[mask1]**2)
                    binsqrsum_x0_carl = np.sum(w_carl[mask0]**2)
                    binsqrsum_x0 = math.sqrt(binsqrsum_x0)
                    binsqrsum_x1 = math.sqrt(binsqrsum_x1)
                    binsqrsum_x0_carl = math.sqrt(binsqrsum_x0_carl)
                    # Form residue
                    res_num = np.sum(w1[mask1]) - np.sum(w0[mask0])
                    res_denom = math.sqrt(binsqrsum_x0**2 + binsqrsum_x1**2)
                    # Form residue (CARL)
                    res_num_carl = np.sum(w1[mask1]) - np.sum(w_carl[mask0])
                    res_denom_carl = math.sqrt(binsqrsum_x0_carl**2 + binsqrsum_x1**2)
                    # Form relative error
                    binsqrsum_x0 = binsqrsum_x0/w0[mask0].sum()
                    binsqrsum_x1 = binsqrsum_x1/w1[mask1].sum()

                    # Save residual
                    x0_error.append(binsqrsum_x0 if binsqrsum_x0 > 0 else 0.0)
                    x1_error.append(binsqrsum_x1 if binsqrsum_x1 > 0 else 0.0)
                    residue.append(res_num/res_denom if binsqrsum_x0+binsqrsum_x1 > 0 else 0.0)
                    residue_carl.append(res_num_carl/res_denom_carl if binsqrsum_x0_carl+binsqrsum_x1 > 0 else 0.0)
                    #print("binsqrsum_x0:  {}".format(binsqrsum_x0))
                    #print("binsqrsum_x1:  {}".format(binsqrsum_x1))
                    #print("residue     :  {}".format(res_num/res_denom))


            # Convert error lists to numpy arrays
            x0_error = np.array(x0_error)
            x1_error = np.array(x1_error)
            residue  = np.array(residue)
            residue_carl  = np.array(residue_carl)

            ## Ratio error
            print("ratio")
            #axes[1].step( xref, yref, where="post", label=legend+" / "+legend, **hist_settings_alt )
            axes[1].step( xref, yref, where="post", **hist_settings_alt )
            axes[1].step( edge1[:-1], x1_ratio, where="post", label="nom / "+legend, **hist_settings_nom)
            axes[1].step( edgecarl[:-1], carl_ratio, where="post", label = '(nominal*CARL) / '+legend, **hist_settings_CARL_ratio)
            axes[1].grid(axis='x', color='silver')
            #plt.fill_between(edge[:,-1], 1.0, x1_ratio)
            yref_error = np.ones(len(edge1))
            yref_error_up = 2* np.sqrt( np.power(x1_error,2) + np.power(x0_error, 2)) # height from bottom
            yref_error_down = yref_error - np.sqrt(np.power(x1_error, 2) + np.power(x0_error,2))
            #print("edg1:    {}".format(edge1))
            #print("yref_error:    {}".format(yref_error))
            #print("x0_error:    {}".format(x0_error))
            #print("x1_error:    {}".format(x1_error))
            #print("width:       {}".format(np.diff(edge1)))
            
            #plt.bar( x=edge1, height=yref_error+x1_error, bottom = yref_error-x1_error, width=np.diff(edge1), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1, label='uncertainty band')
            axes[1].bar( x=edge1[:-1],
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
            
            axes[1].set_ylabel("Ratio", horizontalalignment='center',x=1)
            axes[1].set_xlabel('%s'%(column), horizontalalignment='right',x=1)
            axes[1].legend(frameon=False, ncol=2)#,title = '%s sample'%(label) ) # We want two columns, and the uncertainty band is in the 2nd column
            #axes_1 = plt.gca()
            axes[1].set_ylim([0.5, 1.6])
            axes[1].set_yticks(np.arange(0.5,1.6,0.1))
            #plt.savefig(f"{output_name}_ratio.png")
            #plt.clf()
            #plt.close()


            ## Residual
            print("residual")
            yref = [0.0,0.0]
            ref = axes[2].step( xref, yref, where="post", label=legend+" / "+legend, **hist_settings_alt )
            nom = axes[2].step( edge1, residue, where="post", label="nom / "+legend, **hist_settings_nom)
            carl = axes[2].step( edgecarl, residue_carl, where="post", label = '(nominal*CARL) / '+legend, **hist_settings_CARL_ratio)
            axes[2].grid(axis='x', color='silver')
            #plt.fill_between(edge[:,-1], 1.0, x1_ratio)
            yref_error = np.zeros(len(edge1))
            yref_error_up = np.full(len(edge1), 1)
            yref_error_down = np.full(len(edge1), -1)
            yref_3error_up = np.full(len(edge1), 3)
            yref_3error_down = np.full(len(edge1), -3)
            yref_5error_up = np.full(len(edge1), 5)
            yref_5error_down = np.full(len(edge1), -5)
            #print("edg1:    {}".format(edge1))
            #print("yref_error:    {}".format(yref_error))
            #print("x0_error:    {}".format(x0_error))
            #print("x1_error:    {}".format(x1_error))
            #print("width:       {}".format(np.diff(edge1)))

            #plt.bar( x=edge1, height=yref_error+x1_error, bottom = yref_error-x1_error, width=np.diff(edge1), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1, label='uncertainty band')
            FiveSigma = axes[2].fill_between(edge1, yref_5error_down, yref_5error_up, color='lightcoral', alpha=0.5, label = "5$\sigma$")
            ThreeSigma = axes[2].fill_between(edge1, yref_3error_down, yref_3error_up, color='bisque', alpha=0.75, label = "3$\sigma$")
            OneSigma = axes[2].fill_between(edge1, yref_error_down, yref_error_up, color='olivedrab', alpha=0.5, label = "1$\sigma$")
            
            axes[2].set_ylabel("Residual", horizontalalignment='center',x=1)
            axes[2].set_xlabel('%s'%(column), horizontalalignment='right',x=1)
            axes[2].legend(frameon=False,
                           ncol=3,
                           #title = '%s sample'%(label), 
                           handles=[OneSigma,ThreeSigma,FiveSigma],#,ref,nom,carl], 
                           labels = ["1$\sigma$", "3$\sigma$", "5$\sigma$"])#,("{} / {}").format(legend,legend), ("nom / {}").format(legend),("(nominal*CARL) / {}").format(legend)] )
            #axes = plt.gca()
            axes[2].set_ylim([-8, 8])
            axes[2].set_yticks(np.arange(-8,8,1.0))
            #fig.savefig(f"{output_name}_residual.png")
            fig.savefig(f"{output_name}.png")
            fig.clf()
            axes = [a_ele.clear() for a_ele in axes]
            #fig.close()
            
            #if id > 1:
            #    return
            
def draw_resampled_ratio(x0, w0, x1, w1, ratioName=''):

    # Determine bins from (potentially) nan present data
    bins = np.linspace(np.amin(x0), np.amax(x0) ,50)

    # Protection for nans - bad approach as it over writes the original data
    x0 = np.nan_to_num(x0, nan=np.inf) # set to infinite, thereby removing the dummy value from the plotting range
    x1 = np.nan_to_num(x1, nan=np.inf) # set to infinite, thereby removing the dummy value from the plotting range
    print("x0 :  {}".format(x0))
    print("x1 :  {}".format(x1))
    print("w0 :  {}".format(w0))
    print("w1 :  {}".format(w1))

    n0, _, _ = plt.hist(x0, weights=w0, bins=bins, label='original', **hist_settings_nom)
    n1, _, _ = plt.hist(x1, weights=w1, bins=bins, label='resampled', **hist_settings_alt)
    plt.savefig(f'plots/subsample_raw_{ratioName}.png')
    plt.clf()

    print("n0 :  {}".format(n0))
    print("n1 :  {}".format(n1))

    ratio = [n1i/n0i-1 if n0i!=0 else -1 for (n0i,n1i) in zip(n0,n1)]

    #error
    x0_error = []
    x1_error = []
    width = abs(bins[1] - bins[0] )
    for xbin in bins:
        # Form masks for all event that match condition
        mask0 = (x0 < (xbin + width)) & (x0 > (xbin - width))
        mask1 = (x1 < (xbin + width)) & (x1 > (xbin - width))
        # Form bin error
        binsqrsum_x0 = np.sum(w0[mask0]**2)
        binsqrsum_x1 = np.sum(w1[mask1]**2)
        binsqrsum_x0 = math.sqrt(binsqrsum_x0)
        binsqrsum_x1 = math.sqrt(binsqrsum_x1)
        ratio_binsqrsum_x0 = binsqrsum_x0
        # Form relative error
        binsqrsum_x0 = binsqrsum_x0/w0[mask0].sum()
        binsqrsum_x1 = binsqrsum_x1/w1[mask1].sum()

        x0_error.append(binsqrsum_x0 if binsqrsum_x0 > 0 else 0.0)
        x1_error.append(binsqrsum_x1 if binsqrsum_x1 > 0 else 0.0)

    # Convert error lists to numpy arrays
    x0_error = np.array(x0_error)
    x1_error = np.array(x1_error)

    bin_centers = [ (bins[i+1]+bins[i])/2 for i in range(len(bins)-1)]
    ones = [1 for b in bin_centers]
    plt.hist(bin_centers, weights=ratio, bins=bins, bottom=ones, label='ratio', histtype='step') #, **hist_settings_CARL_ratio)

    yref_error = np.ones(len(bins))
    yref_error_up = 2* np.sqrt( np.power(x1_error,2) + np.power(x0_error, 2)) # height from bottom
    yref_error_down = yref_error - np.sqrt(np.power(x1_error, 2) + np.power(x0_error,2))


    plt.bar( x=bins[:-1],
             height=yref_error_up[:-1], bottom = yref_error_down[:-1],
             color='red',
             width=np.diff(bins),
             align='edge',
             alpha=0.25,
             label='uncertainty band')

    plt.ylabel('resampled / original')
    plt.legend(frameon=False )
    axes = plt.gca()
    plt.savefig(f'plots/ratio_{ratioName}.png')
    plt.clf()
    plt.close()

def subsample(x0, w0, label, nEvents, globalName, featureNames=[]):
    
    w0 = w0.flatten()
    
    # Remove negative probabilities - maintains proportionality still by abs()
    w0_abs = abs(w0)

    # Dataset 0 probability proportionality sub-sampling
    x0_len = x0.shape[0]
    w0_abs_sum = int(w0_abs.sum())
    w0_abs = w0_abs / w0_abs.sum()
    weighted_data0 = np.random.choice(range(x0_len), nEvents, p = w0_abs)
    subsampled_x0 = x0.copy()[weighted_data0]
    
    # set of +-1 weights, depending on the sign of the original weight
    w0_ones = np.ones(x0_len)
    w0_ones[w0<0] = -1
    subsampled_w0 = w0_ones.copy()[weighted_data0]
    
    #===========================================================================
    for n, feature in enumerate(featureNames):
        draw_resampled_ratio(x0[:,n], w0/np.sum(w0), subsampled_x0[:,n], subsampled_w0/np.sum(subsampled_w0), "_subsample_{}_{}".format(globalName,feature))
    #===========================================================================
    
    # Create the labels
    subsampled_y0 = np.full(subsampled_x0.shape, label)

    return (subsampled_x0, subsampled_w0, subsampled_y0)
    
def weight_obs_data(x0, x1, w0, w1, ratioName=''):

    # Remove negative probabilities - maintains proportionality still by abs()
    w0_abs = abs(w0)
    w1_abs = abs(w1)

    # Dataset 0 probability proportionality sub-sampling
    x0_len = x0.shape[0]
    w0_abs_sum = int(w0_abs.sum())
    w0_abs = w0_abs / w0_abs.sum()
    weighted_data0 = np.random.choice(range(x0_len), w0_abs_sum, p = w0_abs)
    w_x0 = x0.copy()[weighted_data0]

    # set of +-1 weights, depending on the sign of the original weight
    w0_ones = np.ones(x0_len)
    w0_ones[w0<0] = -1
    w_w0 = w0_ones.copy()[weighted_data0]

    # Dataset 1 probability proportionality sub-sampling
    x1_len = x1.shape[0]
    w1_abs_sum = int(w1_abs.sum())
    w1_abs = w1_abs / w1_abs.sum()
    weighted_data1 = np.random.choice(range(x1_len), w1_abs_sum, p = w1_abs)
    w_x1 = x1.copy()[weighted_data1]

    # set of +-1 weights, depending on the sign of the original weight
    w1_ones = np.ones(x1_len)
    w1_ones[w1<0] = -1
    w_w1 = w1_ones.copy()[weighted_data1]

    # Concatenate all data
    x_all = np.append(w_x0,w_x1)
    y_all = np.zeros(w0_abs_sum+w1_abs_sum)
    y_all[w0_abs_sum:] = 1
    w_all = np.concatenate([w_w0, w_w1])

    #===========================================================================
    if ratioName!='':
        draw_resampled_ratio(x0, w0, w_x0, w_w0, ratioName+'_x0')
        draw_resampled_ratio(x1, w1, w_x1, w_w1, ratioName+'_x1')
    #===========================================================================

    ## no resampling
    # x_all = np.append(x0,x1)
    # y_all = np.zeros(x0.shape[0]+x1.shape[0])
    # y_all[x0.shape[0]:] = 1
    # w_all = np.concatenate([w0, w1])

    return (x_all,y_all,w_all)

def obs_roc_curve(x, y_true, weights):

    # Determine the maximum range
    #maxRange = np.amax(x)
    maxRange = np.percentile(x, 99)
    #minRange = np.amin(x)
    minRange = np.amin(x, 0)
    print("       -> Range:  {},{}".format(maxRange,minRange))

    # Now determine the boundaries for classification
    ClassBoundaries = np.linspace(minRange, maxRange, 20)

    # Default ROC curve info
    fpr = np.zeros(len(ClassBoundaries))
    tpr = np.zeros(len(ClassBoundaries))

    # Loop through boundaries and determine predicted confusion matrix values
    for idx,edge in enumerate(ClassBoundaries):
        tpr[idx] = 0.0
        fpr[idx] = 0.0
        # print("-> Edge:  {}".format(idx))
        y_pred =  x < edge
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, sample_weight=weights).ravel()
        tpr[idx] = tp/(tp+fn)
        fpr[idx] = fp/(tn+fp)
        #print("       -> tpr:  {}".format(tpr))
        #print("       -> fpr:  {}".format(fpr))

    return fpr, tpr

def resampled_obs_and_roc(original, target, w0, w1, ratioName=''):

    (data, labels, weights) = weight_obs_data(original, target, w0, w1, ratioName)
    fpr, tpr  = obs_roc_curve(data, labels, weights)
    #roc_auc = auc(fpr, tpr)
    roc_auc = np.trapz(tpr, x=fpr)

    return fpr,tpr,roc_auc,data,labels,weights

def draw_Obs_ROC(X0, X1, W0, W1, weights, label, legend, n, plot = True, plot_resampledRatio = False):
    plt.figure(figsize=(8, 6))
    W0 = W0.flatten()
    W1 = W1.flatten()
    for idx in range(X0.shape[1]): ## Loop through the observables and calculate ROC
        print("<draw_Obs_ROC()>::   Observable {}".format(idx))
        # Extract the observables - 1D array
        x0 = X0[:,idx]
        x1 = X1[:,idx]

        # Form the resampled data based on probability of each event
        # file names for the ratios
        ratioNameNominal = 'nom_alt' if plot_resampledRatio else ''
        ratioNameWeighted = 'carlW_alt' if plot_resampledRatio else ''
        #
        fpr_t,tpr_t,roc_auc_t,data_t,labels_t,weights_t = resampled_obs_and_roc(x0, x1, W0, W1, ratioName=ratioNameNominal)
        fpr_tC,tpr_tC,roc_auc_tC,data_tr,labels_tr,weights_tr = resampled_obs_and_roc(x0, x1, W0*weights, W1, ratioName=ratioNameWeighted)
        plt.plot(fpr_t, tpr_t, label=r"no weight, AUC=%.3f" % roc_auc_t)
        plt.plot(fpr_tC, tpr_tC, label=r"CARL weight, AUC=%.3f" % roc_auc_tC)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Resampled proportional to weights (obs: {})'.format(idx))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right", title = label)
        plt.tight_layout()
        if plot:
            plt.savefig('plots/roc_nominalVs%s_%s_%s_%s.png'%(legend,label,idx,n))
            plt.clf()
            logger.info("   Observable %s"%(idx))
            logger.info("CARL weighted %s AUC is %.3f"%(label,roc_auc_tC))
            logger.info("Unweighted %s AUC is %.3f"%(label,roc_auc_t))
            logger.info("Saving ROC plots to /plots")

        # Plot variables used in ROC calculation
        bins = np.linspace(np.amin(x0), np.amax(x0) ,50)
        plt.hist(data_t[labels_t==0],   bins=bins, weights=weights_t[labels_t==0], label=r"Nominal", **hist_settings_nom)
        plt.hist(data_tr[labels_tr==0], bins=bins, weights=weights_tr[labels_tr==0], label=r"Nominal * CARL", **hist_settings_CARL)
        plt.hist(data_tr[labels_tr==1], bins=bins, weights=weights_tr[labels_tr==1], label=r"Alternative", **hist_settings_alt)
        plt.title('Resampled proportional to weights (obs: {})'.format(idx))
        plt.xlabel('Obseravble {}'.format(idx))
        plt.ylabel('Events')
        plt.legend(loc="lower right", title = label)
        plt.tight_layout()
        if plot:
            plt.savefig('plots/roc_inputs_nominalVs%s_%s_%s_%s.png'%(legend,label,idx,n))
            plt.clf()

def weight_data(x0, x1, w0, w1):

    # Remove negative probabilities - maintains proportionality still by abs()
    w0 = abs(w0)
    w1 = abs(w1)

    # Test the amount of available memory on device
    mem = psutil.virtual_memory()
    # Assign a max resample number
    max_resample = 200000
    
    x0_len = x0.shape[0]
    w0_sum = int(w0.sum())
    print("<weight_data>::   Initial resample length (w0_sum): {}".format(w0_sum))
    print("<weight_data>::   Available virtual memory  = {}".format(mem.free))
    print("<weight_data>::   Size of single data point = {}".format(x0[0].nbytes))
    print("<weight_data>::      ->  x0[0] = {}".format(x0[0]))
    print("<weight_data>::      ->  x1[0] = {}".format(x1[0]))
    resample_num_0 = round(mem.free/(10*x0[0].nbytes))
    resample_num_0 = resample_num_0 if resample_num_0 < max_resample else max_resample
    w0 = w0 / w0.sum()
    #weighted_data0 = np.random.choice(range(x0_len), w0_sum, p = w0)
    weighted_data0 = np.random.choice(range(x0_len), resample_num_0, p = w0)
    print("<weight_data>::    Weighted Sample Length (x0) = {}".format(len(weighted_data0)))
    w_x0 = x0.copy()[weighted_data0]

    x1_len = x1.shape[0]
    w1_sum = int(w1.sum())
    resample_num_1 = round(mem.free/(10*x1[0].nbytes))
    resample_num_1 = resample_num_1 if resample_num_1 < max_resample else max_resample
    resample_num_1 *= (w1_sum/w0_sum)
    resample_num_1 = round(resample_num_1)
    w1 = w1 / w1.sum()
    #weighted_data1 = np.random.choice(range(x1_len), w1_sum, p = w1)
    weighted_data1 = np.random.choice(range(x1_len), resample_num_1, p = w1)
    print("<weight_data>::    Weighted Sample Length (x1) = {}".format(len(weighted_data1)))
    w_x1 = x1.copy()[weighted_data1]

    # Calculate the minimum size so as to ensure we have equal number of events in each class
    minEvts = min([len(w_x0),len(w_x1)])
    w_x0 = w_x0[ 0:minEvts, :]
    w_x1 = w_x1[ 0:minEvts, :]

    x_all = np.vstack((w_x0,w_x1))
    y_all = np.zeros(2*len(w_x0))
    y_all[len(w_x0):] = 1

    return (x_all,y_all)

def resampled_discriminator_and_roc(original, target, w0, w1):
    (data, labels) = weight_data(original, target, w0, w1)
    Xtr, Xts, Ytr, Yts = train_test_split(data, labels, random_state=42, train_size=0.51, test_size=0.49)

    discriminator = MLPRegressor(tol=1e-05, activation="logistic",
                                 hidden_layer_sizes=(original.shape[1],original.shape[1], original.shape[1]),
                                 learning_rate_init=1e-07, learning_rate="constant",
                                 solver="lbfgs", random_state=1,
                                 max_iter=200)

    discriminator.fit(Xtr,Ytr)
    predicted = discriminator.predict(Xts)
    fpr, tpr, _  = roc_curve(Yts,predicted.ravel())
    roc_auc = auc(fpr, tpr)
    return fpr,tpr,roc_auc

def draw_ROC(X0, X1, W0, W1, weights, label, legend, n, plot = True):
    plt.figure(figsize=(8, 6))
    W0 = W0.flatten()
    W1 = W1.flatten()
    fpr_t,tpr_t,roc_auc_t = resampled_discriminator_and_roc(X0, X1, W0, W1)
    fpr_tC,tpr_tC,roc_auc_tC = resampled_discriminator_and_roc(X0, X1, W0*weights, W1)
    plt.plot(fpr_t, tpr_t, label=r"no weight, AUC=%.3f" % roc_auc_t)
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

def plot_calibration_curve(y, probs_raw, probs_cal, global_name, save = False):
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Masking -ve probabilities
    mask = np.logical_and( (probs_raw > 0), (probs_cal > 0))
    y = y[mask]
    probs_raw = probs_raw[mask]
    probs_cal = probs_cal[mask]
    #for idx,(i,j) in enumerate(zip(probs_raw, probs_cal)):
    #    if i < 0:
    #        probs_raw[idx] = y[idx]
    #    if j < 0:
    #        probs_cal[idx] = y[idx]
    #    #if j < 0 or i < 0:
    #        #print("probs_raw:   {}".format(i))
    #        #print("probs_cal:   {}".format(j))



    frac_of_pos_raw, mean_pred_value_raw = calibration_curve(y, probs_raw, n_bins=50)#, normalize=True)
    frac_of_pos_cal, mean_pred_value_cal = calibration_curve(y, probs_cal, n_bins=50)#, normalize=True)

    ax1.plot(mean_pred_value_raw, frac_of_pos_raw, "s-", label='uncalibrated', **hist_settings_nom)
    ax1.plot(mean_pred_value_cal, frac_of_pos_cal, "s-", label='calibrated', **hist_settings_alt)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration plot')

    ax2.hist(probs_raw, range=(0, 1), bins=50, label='uncalibrated', lw=2, **hist_settings_nom)
    ax2.hist(probs_cal, range=(0, 1), bins=50, label='calibrated', lw=2, **hist_settings_alt)
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    if save:
        plt.savefig('plots/calibration_'+global_name+'.png')
        plt.clf()
    logger.info("Saving calibration curves to /plots")

def draw_weights(weightCT, weightCA, legend, do, n, save = False):
    plt.yscale('log')
    plt.hist(weightCT, bins = np.exp(np.linspace(-0.5,1.1,50)), label = 'carl-torch', **hist_settings_nom)
    plt.hist(weightCA, bins = np.exp(np.linspace(-0.5,1.1,50)), label = 'carlAthena', **hist_settings_nom)
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


def compute_probs(data, weights, n=100): 
    h, e = np.histogram(data, weights=weights, bins=n)
    p = h/np.sum(weights) #data.shape[0]
    return e, p

def support_intersection(p, q): 
    sup_int = (
        list(
            filter(
                lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
            )
        )
    )
    return sup_int

def get_probs(list_of_tuples): 
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

def kl_divergence(p, q): 
    return np.sum(p*np.log(p/q))

def compute_kl_divergence(train_sample, train_weights, test_sample, test_weights, n_bins=10): 
    """
    Computes the KL Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, train_weights, n=n_bins)
    _, q = compute_probs(test_sample, test_weights, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)
    
    return kl_divergence(p, q)
