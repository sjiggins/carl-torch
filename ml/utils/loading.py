from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import logging
import tarfile
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict
from .tools import create_missing_folders, load, load_and_check, HarmonisedLoading
from .plotting import draw_weighted_distributions, draw_unweighted_distributions, draw_ROC, draw_Obs_ROC, resampled_obs_and_roc, plot_calibration_curve, draw_weights, draw_scatter
from sklearn.model_selection import train_test_split
import yaml
logger = logging.getLogger(__name__)


class Loader():
    """
    Loading of data.
    """
    def __init__(self):
        super(Loader, self).__init__()

    def loading(
        self,
        folder=None,
        plot=False,
        global_name="Test",
        features=[],
        weightFeature="DummyEvtWeight",
        TreeName = "Tree",
        x0 = None,
        x1 = None,
        randomize = False,
        save = False,
        correlation = True,
        preprocessing = False,
        nentries = 0,
        pathA = '',
        pathB = '',
        normalise = False,
        debug = False,
        noTar = True,
    ):
        """
        Parameters
        ----------
        folder : str or None
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format). Default value:            None.
        plot : bool, optional
            make validation plots
        global_name : str
            Name of containing folder for executed training or evaluation run
        do : str
            Decide what samples to use. Can either be Sherpa Vs Madgraph ('sherpaVsMG5'), Renormalization scale up vs down ('mur') or qsf scale up vs down ('qsf')
            Default value: 'sherpaVsMG5'
        x0 : dataframe of none
            Either pass a dataframe as in notebook, or None to load sample according to do option.
        x1 : dataframe of none
            Either pass a dataframe as in notebook, or None to load sample according to do option.
        randomize : bool, optional
            Randomize training sample. Default value:
            False
        save : bool, optional
            Save training ans test samples. Default value:
            False
        Returns
        -------
        x : ndarray
            Observables with shape `(n_samples, n_observables)`. The same information is saved as a file in the given
            folder.
        y : ndarray
            Class label with shape `(n_samples, n_parameters)`. `y=0` (`1`) for events sample from the numerator
            (denominator) hypothesis. The same information is saved as a file in the given folder.
        """

        # Create folders for storage
        create_missing_folders([folder+'/'+global_name])
        create_missing_folders(['plots'])

        # Extract the TTree data as pandas dataframes
        (
            x0, w0, vlabels0,
            x1, w1, vlabels1
        )  = HarmonisedLoading(fA = pathA, fB = pathB,
                               features=features, weightFeature=weightFeature,
                               nentries = int(nentries), TreeName = TreeName)

        # Run if requested debugging by user
        if debug:
            print("<loading.py::Loader()>::   Data sets for training (pandas dataframe)")
            print("<loading.py::Loader()>::      X0:")
            print(x0)
            print("<loading.py::Loader()>::      X1:")
            print(x1)

        # Diferent filters of observables to keep only the interesting ones (provisional)
        feat_interest = ["Jet1_lund_z0","Jet1_lund_z1","Jet1_lund_z2","Jet1_lund_z3",
                         "Jet2_lund_z0","Jet2_lund_z1","Jet2_lund_z2","Jet2_lund_z3",
                         "Jet1_pTrel_ch0","Jet1_pTrel_ch1","Jet1_pTrel_ch2",
                         "Jet2_pTrel_ch0","Jet2_pTrel_ch1","Jet2_pTrel_ch2",
                         "Jet3_pTrel_ch0","Jet3_pTrel_ch1","Jet3_pTrel_ch2",
                         "Jet1_JetShape_ch0","Jet1_JetShape_ch1","Jet1_JetShape_ch2",
                         "Jet2_JetShape_ch0","Jet2_JetShape_ch1","Jet2_JetShape_ch2",
                         "Jet3_JetShape_ch0","Jet3_JetShape_ch1","Jet3_JetShape_ch2",
                         "Ntracks",
                         "N_ch0","N_ch1","N_ch2",
                         "Jet_Mass0","Jet_Mass1","Jet_Mass2",
                         "JetCharge0","JetCharge1","JetCharge2",
                         "ptFracB0","ptFracB1","ptFracB2",
                         "ptFracC0","ptFracC1","ptFracC2"]

        feat_interestB = ["Jet1_lund_z1","Jet1_lund_z2","Jet1_lund_z3",
                         "Jet1_lund_dR1","Jet1_lund_dR2","Jet1_lund_dR3",
                         "Jet1_track_Pt0","Jet1_track_Pt1","Jet1_track_Pt2",
                         "Jet2_lund_z1","Jet2_lund_z2","Jet2_lund_z3",
                         "Jet2_lund_dR1","Jet2_lund_dR2","Jet2_lund_dR3",
                         "Jet2_track_Pt0","Jet2_track_Pt1","Jet2_track_Pt2",
                         "Jet_Mass0","Jet_Mass1",
                         "JetCharge0","JetCharge1",
                         "ptFracC0","ptFracC1",
                         "ptFracB0","ptFracB1",
                         "Ntracks",
                         "N_ch0","N_ch1",
                         "Jet1_z_ch0","Jet1_z_ch1","Jet1_z_ch2",
                         "Jet1_pTrel_ch0","Jet1_pTrel_ch1","Jet1_pTrel_ch2",
                         "Jet1_JetShape_ch0","Jet1_JetShape_ch1","Jet1_JetShape_ch2",
                         "Jet2_z_ch0","Jet2_z_ch1","Jet2_z_ch2",
                         "Jet2_pTrel_ch0","Jet2_pTrel_ch1","Jet2_pTrel_ch2",
                         "Jet2_JetShape_ch0","Jet2_JetShape_ch1","Jet2_JetShape_ch2"]

        feat_interestC = ["Jet1_lund_z0","Jet1_lund_z1","Jet1_lund_z2","Jet1_lund_z3",
                         "Jet2_lund_z0","Jet2_lund_z1","Jet2_lund_z2","Jet2_lund_z3",
                         "Jet1_pTrel_ch0","Jet1_pTrel_ch1","Jet1_pTrel_ch2",
                         "Jet2_pTrel_ch0","Jet2_pTrel_ch1","Jet2_pTrel_ch2",
                         "Jet3_pTrel_ch0","Jet3_pTrel_ch1","Jet3_pTrel_ch2",
                         "Ntracks",
                         "N_ch0","N_ch1","N_ch2",
                         "Jet_Mass0","Jet_Mass1","Jet_Mass2",
                         "Jet_Pt0","Jet_Pt1","Jet_Pt2",
                         "Jet_Eta0","Jet_Eta1","Jet_Eta2",
                         "JetCharge0","JetCharge1","JetCharge2"]

        feat_interestD = ["Jet1_lund_z0",
                         "Jet2_lund_z0",
                         "Jet1_pTrel_ch0",
                         "Jet2_pTrel_ch0",
                         "Jet3_pTrel_ch0",
                         "Ntracks",
                         "N_ch0","N_ch1","N_ch2",
                         "Jet_Mass0","Jet_Mass1","Jet_Mass2",
                         "JetCharge0","JetCharge1","JetCharge2"]

        feat_interestE = ["Ntracks",
                          "Jet_Mass0","Jet_Mass1",
                          "JetCharge0","JetCharge1",
                          "N_ch0", "N_ch1",
                          "ptFracB0","ptFracB1",
                          "ptFracC0","ptFracC1"]

        initial_columns = x0.columns
        for f in initial_columns:#No loop and delete in same list
            logger.info("Observable column {}, Dummy porportion {}".format(f, np.isnan(x0[f]).sum()/len(x0[f])))
            if not f in feat_interest:
                del x0[f]
                del x1[f]
                logger.info(" Deleting observable column {}".format(f))
        #    else:
                # Drop columns with large amount of dummy values
#            logger.info("Observable column {}, Dummy porportion {}".format(f, np.isnan(x0[f]).sum()/len(x0[f])))
#            if ((np.isnan(x0[f])).sum()/len(x0[f]) > 0.5):
#                logger.info("Deleting observable column {} with more than 50% of dummy values".format(f))
#                del x0[f]
#                del x1[f]
        logger.info("Final list of observables: {}".format(x0.columns))

        # Pre-process for outliers
        logger.info(" Starting filtering")
        if preprocessing:
            factor = 5 # 5 sigma deviation
            x00 = len(x0)
            x10 = len(x1)
            for column in x0.columns:
                upper_lim0 = x0[column].mean () + x0[column].std () * factor
                upper_lim1 = x1[column].mean () + x1[column].std () * factor
                lower_lim0 = x0[column].mean () - x0[column].std () * factor
                lower_lim1 = x1[column].mean () - x1[column].std () * factor
                upper_lim = upper_lim0 if upper_lim0 > upper_lim1 else upper_lim1
                lower_lim = lower_lim0 if lower_lim0 < lower_lim1 else lower_lim1

                # If the std = 0, then skip as this is a singular value feature
                #   Can happen during zero-padding
                if x0[column].std == 0 or x1[column].std () == 0:
                    continue

                if debug:
                    print("Column: {}:".format(column))
                    print("Column: {},  mean0 = {}".format(column, x0[column].mean ()))
                    print("Column: {},  mean1 = {}".format(column, x1[column].mean ()))
                    print("Column: {},  std0 = {}".format(column, x0[column].std ()))
                    print("Column: {},  std1 = {}".format(column, x1[column].std ()))
                    print("Column: {},  lower limit = {}".format(column,lower_lim))
                    print("Column: {},  upper limit = {}".format(column,upper_lim))
                x0_mask = (x0[column] < upper_lim) & (x0[column] > lower_lim)
                x1_mask = (x1[column] < upper_lim) & (x1[column] > lower_lim)

                x0 = x0[x0_mask]
                x1 = x1[x1_mask]

                # Filter weights
                w0 = w0[x0_mask]
                w1 = w1[x1_mask]
            x0 = x0.round(decimals=2)
            x1 = x1.round(decimals=2)

            if debug:
                logger.info(" Filtered x0 outliers in percent: %.2f", (x00-len(x0))/len(x0)*100)
                logger.info(" Filtered x1 outliers in percent: %.2f", (x10-len(x1))/len(x1)*100)
                logger.info("weight vector (0): {}".format(w0))
                logger.info("weight vector (1): {}".format(w1))

        if correlation:
            cor0 = x0.corr()

            # Save in txt file
            logger.info("Correlation Matrix: {}".format(cor0))
            f_corr = open("Corr/corr_"+global_name+".txt", "w")
            corr_string = cor0.to_string(header=True,index=True)
            f_corr.write(corr_string)
            cor2 = cor0[abs(cor0)>0.15]
            f_corr2 = open("Corr/corr2_"+global_name+".txt", "w")
            corr2_string = cor2.to_string(header=True,index=True)
            f_corr2.write(corr2_string)

            # corr plot
            sns.heatmap(cor0, annot=True, cmap=plt.cm.Reds)
            cor_target = abs(cor0[x0.columns[0]])
            relevant_features = cor_target[cor_target>0.15]
            if plot:
                plt.savefig('plots/scatterMatrix_'+global_name+'.png')
                plt.clf()

        #if plot and int(nentries) > 10000: # no point in plotting distributions with too few events
        #    logger.info(" Making plots")
        #    draw_unweighted_distributions(x0.to_numpy(), x1.to_numpy(),
        #                                  np.ones(x0.to_numpy()[:,0].size),
        #                                  x0.columns,
        #                                  vlabels1,
        #                                  binning,
        #                                  global_name,
        #                                  nentries,
        #                                  plot)

        # sort dataframes alphanumerically
        x0 = x0[sorted(x0.columns)]
        x1 = x1[sorted(x1.columns)]

        # get metadata, i.e. max, min, mean, std of all the variables in the dataframes
        metaData = {v : {x0[v].min(), x0[v].max() } for v in  x0.columns } # Try also x0[v].nsmallest(2).iloc[-1] instead of 0.0
        X0 = x0.to_numpy()
        X1 = x1.to_numpy()

        # Convert weights to numpy
        w0 = w0.to_numpy()
        w1 = w1.to_numpy()
        if normalise:
            w0 = w0 / (w0.sum())
            w1 = w1 / (w1.sum())

        # Target labels
        y0 = np.zeros(x0.shape[0])
        y1 = np.ones(x1.shape[0])

        # Train, test splitting of input dataset
        X0_train, X0_test, y0_train, y0_test, w0_train, w0_test = train_test_split(X0, y0, w0, test_size=0.40, random_state=42)
        X1_train, X1_test, y1_train, y1_test, w1_train, w1_test = train_test_split(X1, y1, w1, test_size=0.40, random_state=42)
        X0_train, X0_val,  y0_train, y0_val, w0_train, w0_val =  train_test_split(X0_train, y0_train, w0_train, test_size=0.50, random_state=42)
        X1_train, X1_val,  y1_train, y1_val, w1_train, w1_val =  train_test_split(X1_train, y1_train, w1_train, test_size=0.50, random_state=42)
        X_train = np.vstack([X0_train, X1_train])
        y_train = np.concatenate((y0_train, y1_train), axis=None)
        w_train = np.concatenate( (w0_train, w1_train), axis=None)
        X_val   = np.vstack([X0_val, X1_val])
        y_val = np.concatenate((y0_val, y1_val), axis=None)
        w_val = np.concatenate( (w0_val, w1_val), axis=None)
        X = np.vstack([X0, X1])
        y = np.concatenate((y0, y1), axis=None)
        w = np.concatenate( (w0,w1), axis=None)
        # save data
        if folder is not None and save:
            np.save(folder + global_name + "/X_train_" +str(nentries)+".npy", X_train)
            np.save(folder + global_name + "/y_train_" +str(nentries)+".npy", y_train)
            np.save(folder + global_name + "/w_train_" +str(nentries)+".npy", w_train)
            np.save(folder + global_name + "/X_val_"   +str(nentries)+".npy", X_val)
            np.save(folder + global_name + "/y_val_"   +str(nentries)+".npy", y_val)
            np.save(folder + global_name + "/w_val_"   +str(nentries)+".npy", w_val)
            np.save(folder + global_name + "/X0_val_"  +str(nentries)+".npy", X0_val)
            np.save(folder + global_name + "/X1_val_"  +str(nentries)+".npy", X1_val)
            np.save(folder + global_name + "/w0_val_"  +str(nentries)+".npy", w0_val)
            np.save(folder + global_name + "/w1_val_"  +str(nentries)+".npy", w1_val)
            np.save(folder + global_name + "/X0_train_"+str(nentries)+".npy", X0_train)
            np.save(folder + global_name + "/X1_train_"+str(nentries)+".npy", X1_train)
            np.save(folder + global_name + "/w0_train_"  +str(nentries)+".npy", w0_train)
            np.save(folder + global_name + "/w1_train_"  +str(nentries)+".npy", w1_train)
            f = open(folder + global_name + "/metaData_"+str(nentries)+".pkl", "wb")
            pickle.dump(metaData, f)
            f.close()
            #Tar data files if training is done on GPU
            if torch.cuda.is_available() and not noTar:
                plot = False #don't plot on GPU...
                tar = tarfile.open("data_out.tar.gz", "w:gz")
                for name in [folder + global_name + "/X_train_" +str(nentries)+".npy",
                             folder + global_name + "/y_train_" +str(nentries)+".npy",
                             folder + global_name + "/X_val_"   +str(nentries)+".npy",
                             folder + global_name + "/y_val_"   +str(nentries)+".npy",
                             folder + global_name + "/X0_val_"  +str(nentries)+".npy",
                             folder + global_name + "/X1_val_"  +str(nentries)+".npy",
                             folder + global_name + "/w0_val_"  +str(nentries)+".npy",
                             folder + global_name + "/w1_val_"  +str(nentries)+".npy",
                             folder + global_name + "/X0_train_"+str(nentries)+".npy",
                             folder + global_name + "/X1_train_"+str(nentries)+".npy",
                             folder + global_name + "/w0_train_"  +str(nentries)+".npy",
                             folder + global_name + "/w1_train_"  +str(nentries)+".npy"]:
                    tar.add(name)
                    f = open(folder + global_name + "/metaData_"+str(nentries)+".pkl", "wb")
                    pickle.dump(metaData, f)
                    f.close()
                tar.close()

        return X_train, y_train, X0_train, X1_train, w_train, w0_train, w1_train, metaData



    def load_result(
        self,
        x0,
        x1,
        w0,
        w1,
        metaData,
        weights = None,
        label = None,
        features=[],
        plot = False,
        nentries = 0,
        global_name="Test",
        plot_ROC = True,
        plot_obs_ROC = True,
        ext_binning = None,
        ext_plot_path=None,
        verbose=False,
    ):
        """
        Parameters
        ----------
        weights : ndarray
            r_hat weights:
        Returns
        -------
        """

        if verbose:
            print("<loading.py::load_result>::   Extracting numpy data features")

        # Get data - only needed for column names which we can use features instead
        #x0df, weights0, labels0 = load(f = pathA,
        #                     features=features, weightFeature=weightFeature,
        #                     n = int(nentries), t = TreeName)
        #x1df, weights1, labels1 = load(f = pathB,
        #                     features=features, weightFeature=weightFeature,
        #                     n = int(nentries), t = TreeName)

        # load samples
        X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0)
        X1 = load_and_check(x1, memmap_files_larger_than_gb=1.0)
        W0 = load_and_check(w0, memmap_files_larger_than_gb=1.0)
        W1 = load_and_check(w1, memmap_files_larger_than_gb=1.0)

        if isinstance(metaData, str):
            metaDataFile = open(metaData, 'rb')
            metaDataDict = pickle.load(metaDataFile)
            metaDataFile.close()
        else:
            metaDataDict = metaData
        #weights = weights / weights.sum() * len(X1)

        # Calculate the maximum of each column and minimum and then allocate bins
        if verbose:
            print("<loading.py::load_result>::   Calculating min/max range for plots & binning")
        binning = defaultdict()
        minmax = defaultdict()
        divisions = 50

        # external binning from yaml file.
        if ext_binning:
            with open(ext_binning, "r") as f:
                ext_binning = yaml.load(f, yaml.FullLoader)

        #for idx,column in enumerate(x0df.columns):
        for idx,(key,pair) in enumerate(metaDataDict.items()):

            # check to see if variable is in the yaml file.
            # if not, proceed to automatic binning
            if ext_binning is not None:
                try:
                    binning[idx] = np.arange(*ext_binning["binning"][key])
                    continue
                except KeyError:
                    pass

            #max = x0df[column].max()
            #min = x0df[column].min()
            #minmax[column] = [min,max]
            #binning[column] = np.linspace(min, max, divisions)
            #print("<loading.py::load_result>::   Column {}:  min  =  {},  max  =  {}".format(column,min,max))

            #  Min/max
            #max = np.amax(X0[:,idx])
            #min = np.amin(X0[:,idx])
            #minmax[idx] = [min,max]
            #binning[idx] = np.linspace(min, max, divisions)
            #print("<loading.py::load_result>::   Column {}:  min  =  {},  max  =  {}".format(column,min,max))

            #  Mean/std
            mean = np.nanmean(X0[:,idx]) # mean = np.mean(X0[:,idx]) does not work with NaN values
            std = np.nanstd(X0[:,idx])  # mean = np.std(X0[:,idx]) does not work with NaN values
            #if (key == "Jet1_lund_dR0" or "Jet1_lund_z0" or "Jet2_lund_z0" or "Jet2_lund_dR0"):
            #    std = 2.

            print("<loading.py::load_result>:: Standard deviation to find binning limits:")
            print("     idx = {}, key = {}, pair = {}, sigma = {}".format(idx, key, pair, std))
            factor = 5
            minmax[idx] = [mean-(5*std), mean+(5*std)]
            binning[idx] = np.linspace(mean-(5*std), mean+(5*std), divisions)
            if verbose:
                print("<loading.py::load_result>::   Column {}:  min  =  {},  max  =  {}".format(key,mean-5*std,mean+5*std))
                print(binning[idx])


        # no point in plotting distributions with too few events, they only look bad
        #if int(nentries) > 5000:

        if verbose:
            print("<loading.py::load_result>::   Printing weighted distributions")

        # plot reweighted distributions
        draw_weighted_distributions(X0, X1, W0, W1,
                                    weights,
                                    metaDataDict.keys(),#x0df.columns,
                                    binning,
                                    label,
                                    global_name, nentries, plot, ext_plot_path)

        # plot ROC curves
        print("<loading.py::load_result>::   Printing ROC")

        # The saved numpy datasets have NaN as dummy values. This is not handled in ROC computation.
        # Therefore, we give an arbitrary non physical value to these dummy values when obtaining the ROC
        X0 = np.nan_to_num(X0, nan=-9)
        X1 = np.nan_to_num(X1, nan=-9)

        if plot_ROC:
            draw_ROC(X0, X1, W0, W1, weights, label, global_name, nentries, plot)
        if plot_obs_ROC:
            draw_Obs_ROC(X0, X1, W0, W1, weights, label, global_name, nentries, plot)


    def validate_result(
        self,
        weightCT = None,
        weightCA = None,
        do = 'dilepton',
        var = 'QSFUP',
        plot = False,
        n = 0,
        path = '',
    ):
        """
        Parameters
        ----------
        weightsCT : ndarray
            weights from carl-torch:
        weightsCA : ndarray
            weights from carlAthena:
        Returns
        -------
        """
        # draw histograms comparing weight from carl-torch (weightCT) from weight infered through carlAthena (ca.weight)
        draw_weights(weightCT, weightCA, var, do, n, plot)
        draw_scatter(weightCT, weightCA, var, do, n)

    def load_calibration(
        #self,
        #y_true,
        #p1_raw = None,
        #p1_cal = None,
        #label = None,
        #do = 'dilepton',
        #var = 'QSFUP',
        #plot = False
        self,
        y_true,
        p1_raw,
        p1_cal,
        label = None,
        features=[],
        plot = False,
        global_name="Test"

    ):
        """
        Parameters
        ----------
        y_true : ndarray
            true targets
        p1_raw : ndarray
            uncalibrated probabilities of the positive class
        p1_cal : ndarray
            calibrated probabilities of the positive class
        Returns
        -------
        """

        # load samples
        y_true  = load_and_check(y_true,  memmap_files_larger_than_gb=1.0)

        plot_calibration_curve(y_true, p1_raw, p1_cal, global_name, plot)
