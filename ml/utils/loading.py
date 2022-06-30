from __future__ import absolute_import, division, print_function, unicode_literals
# import os
# import time
import logging
import tarfile
import torch
import pickle
import numpy as np
import tracemalloc
import gc
# import pandas as pd
import seaborn as sns
# from pandas.plotting import scatter_matrix
# import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from functools import partial
from collections import defaultdict,OrderedDict
from .tools import create_missing_folders, load, load_and_check, HarmonisedLoading
from .plotting import draw_weighted_distributions, draw_unweighted_distributions, draw_ROC, draw_Obs_ROC, resampled_obs_and_roc, plot_calibration_curve, draw_weights, draw_scatter
from sklearn.model_selection import train_test_split
import yaml
import copy
logger = logging.getLogger(__name__)


class Loader():
    """
    Loading of data.
    """
    def __init__(self):
        super(Loader, self).__init__()
        self.Filter=None

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
        weight_preprocess = False,
        weight_preprocess_nsigma = 3,
        large_weight_clipping = False,
        large_weight_clipping_threshold = 1e7,
        weight_polarity = False,
        scaling="minmax",
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

        weight_preprocess: bool, optional
            event weigth pre-processing by mapping event weights that are larger N*standard deviation to the mean value.

        large_weight_clipping: bool, optional
            clipping large event weight.

        Returns
        -------
        x : ndarray
            Observables with shape `(n_samples, n_observables)`. The same information is saved as a file in the given
            folder.
        y : ndarray
            Class label with shape `(n_samples, n_parameters)`. `y=0` (`1`) for events sample from the numerator
            (denominator) hypothesis. The same information is saved as a file in the given folder.
        """

        tracemalloc.start()

        def show_memory_usage():
            current, peak = tracemalloc.get_traced_memory()
            logger.info(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        show_memory_usage()

        # Create folders for storage
        create_missing_folders([folder+'/'+global_name])
        create_missing_folders(['plots'])

        show_memory_usage()

        # Extract the TTree data as pandas dataframes
        (
            x0, w0, vlabels0,
            x1, w1, vlabels1
        )  = HarmonisedLoading(fA = pathA, fB = pathB,
                               features=features, weightFeature=weightFeature,
                               nentries = -1, TreeName = TreeName, 
                               weight_polarity=weight_polarity, Filter=self.Filter)


        # Run if requested debugging by user
        if debug:
            logger.info("Data sets for training (pandas dataframe)")
            logger.info("   X0:")
            logger.info(x0)
            logger.info("   X1:")
            logger.info(x1)

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
                    logger.info("Column: {}:".format(column))
                    logger.info("Column: {},  mean0 = {}".format(column, x0[column].mean ()))
                    logger.info("Column: {},  mean1 = {}".format(column, x1[column].mean ()))
                    logger.info("Column: {},  std0 = {}".format(column, x0[column].std ()))
                    logger.info("Column: {},  std1 = {}".format(column, x1[column].std ()))
                    logger.info("Column: {},  lower limit = {}".format(column,lower_lim))
                    logger.info("Column: {},  upper limit = {}".format(column,upper_lim))
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
            sns.heatmap(cor0, annot=True, cmap=plt.cm.Reds)
            cor_target = abs(cor0[x0.columns[0]])
            relevant_features = cor_target[cor_target>0.5]
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

        show_memory_usage()

        # sort dataframes alphanumerically
        x0 = x0[sorted(x0.columns)]
        x1 = x1[sorted(x1.columns)]

        show_memory_usage()

        if nentries > x0.shape[0]:
            logger.info("Fewer entries in the input than selected by the -e flag. Setting -e to -1.")
            nentries = -1

        logger.info("Sampling x0, w0, vlabels0 ...")
        x0 = x0.sample(nentries) if nentries!=-1 else x0.sample(frac=1.0)
        gc.collect()
        w0 = w0.iloc[x0.index]
        gc.collect()
        logger.info("...done.")

        show_memory_usage()

        logger.info("Sampling x1, w1, vlabels1 ...")
        x1 = x1.sample(nentries) if nentries!=-1 else x1.sample(frac=1.0)
        gc.collect()
        w1 = w1.iloc[x1.index]
        gc.collect()
        logger.info("...done.")

        show_memory_usage()

         # get metadata, i.e. max, min, mean, std of all the variables in the dataframes
        #metaData = defaultdict()
        metaData = OrderedDict()
        if scaling == "standard":
            metaData = {v : {x0[v].mean() , x0[v].std() } for v in  x0.columns }
            logger.info("Storing Z0 Standard scaling metadata: {}".format(metaData))
        elif scaling == "minmax":
            #metaData = {v : OrderedDict({x0[v].min() if x0[v].min() < x1[v].min() else x1[v].min(), x0[v].max() if x0[v].max() > x1[v].max() else x1[v].max() } for v in  x0.columns) }
            metaData = {v : (x0[v].min() if x0[v].min() < x1[v].min() else x1[v].min(), x0[v].max() if x0[v].max() > x1[v].max() else x1[v].max())  for v in  x0.columns }
            for v in x0.columns:
                logger.info("Storing minmax scaling min:: {}".format( x0[v].min() if x0[v].min() < x1[v].min() else x1[v].min() ))
                logger.info("Storing minmax scaling max: {}".format(  x0[v].max() if x0[v].max() > x1[v].max() else x1[v].max() ))
            logger.info("Storing minmax scaling metadata: {}".format(metaData))
        
        show_memory_usage()
        
        if normalise:
            w0 = w0/(w0[0].sum())
            w1 = w1/(w1[0].sum())

        #create target labels
        y0 = np.zeros(x0.shape[0])
        y1 = np.ones(x1.shape[0])


        # split test/validation/train as 5%/47.5%/47.5%
        logger.info("Splitting data...")
        x0_train = x0[0:int(0.475*x0.shape[0])]
        x0_val = x0[int(0.475*x0.shape[0]):int(0.95*x0.shape[0])]
        x0_test = x0[int(0.95*x0.shape[0]):]
        del x0
        gc.collect()

        y0_train = y0[0:int(0.475*y0.shape[0])]
        y0_val = y0[int(0.475*y0.shape[0]):int(0.95*y0.shape[0])]
        y0_test = y0[int(0.95*y0.shape[0]):]
        del y0
        gc.collect()

        w0_train = w0[0:int(0.475*w0.shape[0])]
        w0_val = w0[int(0.475*w0.shape[0]):int(0.95*w0.shape[0])]
        #w0_test = w0[int(0.95*w0.shape[0]):] #don't need this, it's set to None in the original code anyway
        del w0
        gc.collect()

        
        x1_train = x1[0:int(0.475*x1.shape[0])]
        x1_val = x1[int(0.475*x1.shape[0]):int(0.95*x1.shape[0])]
        x1_test = x1[int(0.95*x1.shape[0]):]
        del x1
        gc.collect()

        y1_train = y1[0:int(0.475*y1.shape[0])]
        y1_val = y1[int(0.475*y1.shape[0]):int(0.95*y1.shape[0])]
        y1_test = y1[int(0.95*y1.shape[0]):]
        del y1
        gc.collect()

        w1_train = w1[0:int(0.475*w1.shape[0])]
        w1_val = w1[int(0.475*w1.shape[0]):int(0.95*w1.shape[0])]
        #w1_test = w0[int(0.95*w1.shape[0]):] #don't need this, it's set to None in the original code anyway
        del w1
        gc.collect()
        logger.info("..done.")

        show_memory_usage()

        #convert all dataframes into numpy arrays
        logger.info("Converting pandas dataframes into numpy arrays...")
        x0_train = x0_train.to_numpy()
        gc.collect()
        x0_val = x0_val.to_numpy()
        gc.collect()
        x0_test = x0_test.to_numpy()
        gc.collect()
        #y0_train = y0_train.to_numpy()
        #gc.collect()
        #y0_val = y0_val.to_numpy()
        #gc.collect()
        #y0_test = y0_test.to_numpy()
        #gc.collect()
        w0_train = w0_train.to_numpy()
        gc.collect()
        w0_val = w0_val.to_numpy()
        gc.collect()
        x1_train = x1_train.to_numpy()
        gc.collect()
        x1_val = x1_val.to_numpy()
        gc.collect()
        x1_test = x1_test.to_numpy()
        gc.collect()
        #y1_train = y1_train.to_numpy()
        #gc.collect()
        #y1_val = y1_val.to_numpy()
        #gc.collect()
        #y1_test = y1_test.to_numpy()
        #gc.collect()
        w1_train = w1_train.to_numpy()
        gc.collect()
        w1_val = w1_val.to_numpy()
        gc.collect()
        logger.info("...done.")

        show_memory_usage()

        #define variables used later:

        X0_train = x0_train
        X0_val = x0_val
        X0_test = x0_test
        X1_train = x1_train
        X1_val = x1_val
        X1_test = x1_test

        
        """
        logger.info("Converting data into numpy arrays...")
        X0 = x0.to_numpy()
        X1 = x1.to_numpy()
        logger.info("... done.")

        show_memory_usage()

        logger.info("Converting weights and labels...")
        # Convert weights to numpy
        w0 = w0.to_numpy()
        w1 = w1.to_numpy()
        if normalise:
            w0 = w0 / (w0.sum())
            w1 = w1 / (w1.sum())
        
        # Target labels
        y0 = np.zeros(x0.shape[0])
        y1 = np.ones(x1.shape[0])
        logger.info("... done.")

        x0 = None
        x1 = None

        gc.collect()

        show_memory_usage()
        
        logger.info("Train/test spliting...")
        # Train, test splitting of input dataset
        X0_train, X0_test, y0_train, y0_test, w0_train, w0_test = train_test_split(X0, y0, w0, test_size=0.05, random_state=42) # what is "w0_test" for? maybe a split size of 0.05 if ok.
        logger.info("..1st done...")
        show_memory_usage()
        X0 = None
        gc.collect()
        logger.info("Garbage collection done.")
        show_memory_usage()
        X1_train, X1_test, y1_train, y1_test, w1_train, w1_test = train_test_split(X1, y1, w1, test_size=0.05, random_state=42)
        logger.info("..2nd done...")
        show_memory_usage()
        X1 = None
        gc.collect()
        logger.info("Garbage collection done.")
        show_memory_usage()
        X0_train, X0_val,  y0_train, y0_val, w0_train, w0_val =  train_test_split(X0_train, y0_train, w0_train, test_size=0.50, random_state=42)
        logger.info("..3rd done...")
        show_memory_usage()
        gc.collect()
        logger.info("Garbage collection done.")
        show_memory_usage()
        X1_train, X1_val,  y1_train, y1_val, w1_train, w1_val =  train_test_split(X1_train, y1_train, w1_train, test_size=0.50, random_state=42)
        logger.info("..4th done...")
        show_memory_usage()
        gc.collect()
        logger.info("Garbage collection done.")
        show_memory_usage()
        logger.info("...splitting done.")

        w0_test = None
        w1_test = None
        """

        #cliping large weights, and replace it by 1.0
        raw_w0_train = None
        raw_w1_train = None
        raw_w0_val = None
        raw_w1_val = None
        if large_weight_clipping or weight_preprocess:
            raw_w0_train = copy.deepcopy(w0_train)
            raw_w1_train = copy.deepcopy(w1_train)
            raw_w0_val = copy.deepcopy(w0_val)
            raw_w1_val = copy.deepcopy(w1_val)
            raw_w0_train_sum = w0_train.sum()
            raw_w1_train_sum = w1_train.sum()
            raw_w0_val_sum = w0_val.sum()
            raw_w1_val_sum = w1_val.sum()
            logger.info(f"Training sum w0={raw_w0_train_sum}")
            logger.info(f"Training sum w1={raw_w1_train_sum}")
            logger.info(f"Validation sum w0={raw_w0_val_sum}")
            logger.info(f"Validation sum w1={raw_w1_val_sum}")

        if large_weight_clipping:
            clip_threshold = (-large_weight_clipping_threshold, large_weight_clipping_threshold)
            w0_train = w0_train.clip(*clip_threshold)
            w1_train = w1_train.clip(*clip_threshold)
            w0_val = w0_val.clip(*clip_threshold)
            w1_val = w1_val.clip(*clip_threshold)
            w0_train_sum = w0_train.sum()
            w1_train_sum = w1_train.sum()
            w0_val_sum = w0_val.sum()
            w1_val_sum = w1_val.sum()
            w0_train_per_change = (w0_train_sum - raw_w0_train_sum)/raw_w0_train_sum
            w1_train_per_change = (w1_train_sum - raw_w1_train_sum)/raw_w1_train_sum
            w0_val_per_change = (w0_val_sum - raw_w0_val_sum)/raw_w0_val_sum
            w1_val_per_change = (w1_val_sum - raw_w1_val_sum)/raw_w1_val_sum
            logger.info(f"After large weight clipping, training sum w0={w0_train_sum}, {w0_train_per_change}")
            logger.info(f"After large weight clipping, training sum w1={w1_train_sum}, {w1_train_per_change}")
            logger.info(f"After large weight clipping, validation sum w0={w0_val_sum}, {w0_val_per_change}")
            logger.info(f"After large weight clipping, validation sum w1={w1_val_sum}, {w1_val_per_change}")

        if weight_preprocess:
            w0_train_mean = np.mean(w0_train)
            w1_train_mean = np.mean(w1_train)
            w0_val_mean = np.mean(w0_val)
            w1_val_mean = np.mean(w1_val)
            w0_train_std = np.std(w0_train)
            w1_train_std = np.std(w1_train)
            w0_val_std = np.std(w0_val)
            w1_val_std = np.std(w1_val)

            local_buffers = {
                "w0_train" : {"weight" : w0_train, "mean" : w0_train_mean, "std" : w0_train_std},
                "w1_train" : {"weight" : w1_train, "mean" : w1_train_mean, "std" : w1_train_std},
                "w0_val" : {"weight" : w0_val, "mean" : w0_val_mean, "std" : w0_val_std},
                "w1_val" : {"weight" : w1_val, "mean" : w1_val_mean, "std" : w1_val_std},
            }

            for name, buffer in local_buffers.items():
                simga_above = buffer["mean"] + weight_preprocess_nsigma * abs(buffer["std"])
                simga_below = buffer["mean"] - weight_preprocess_nsigma * abs(buffer["std"])
                m_weight = buffer["weight"]
                sum_w = m_weight.sum()
                m_weight[ m_weight > simga_above] = buffer["mean"]
                m_weight[ m_weight < simga_below] = buffer["mean"]
                per_change = (m_weight.sum() - sum_w)/sum_w
                logger.info(f"After weight preprocessing with {buffer['mean']}, +-{buffer['std']}, training sum {name}={m_weight.sum()}, {per_change}")

        # finalizing dataset format
        X_train = np.vstack([X0_train, X1_train])
        y_train = np.concatenate((y0_train, y1_train), axis=None)
        w_train = np.concatenate((w0_train, w1_train), axis=None)

        # Since we don't pass these 3 variables back, just save it and set to None
        # to avoid too much RAM usage.
        #X_val   = np.vstack([X0_val, X1_val])
        #y_val = np.concatenate((y0_val, y1_val), axis=None)
        #w_val = np.concatenate((w0_val, w1_val), axis=None)

        # X = np.vstack([X0, X1])
        # y = np.concatenate((y0, y1), axis=None)
        # w = np.concatenate((w0, w1), axis=None)

        # save data
        if folder is not None and save:
            np.save(folder + global_name + "/X_train_" +str(nentries)+".npy", X_train)
            np.save(folder + global_name + "/y_train_" +str(nentries)+".npy", y_train)
            np.save(folder + global_name + "/w_train_" +str(nentries)+".npy", w_train)

            X_val = np.vstack([X0_val, X1_val])
            np.save(folder + global_name + "/X_val_"   +str(nentries)+".npy", X_val)
            X_val = None

            y_val = np.concatenate((y0_val, y1_val), axis=None)
            np.save(folder + global_name + "/y_val_"   +str(nentries)+".npy", y_val)
            y_val = None

            w_val = np.concatenate((w0_val, w1_val), axis=None)
            np.save(folder + global_name + "/w_val_"   +str(nentries)+".npy", w_val)
            w_val = None

            np.save(folder + global_name + "/X0_val_"  +str(nentries)+".npy", X0_val)
            np.save(folder + global_name + "/X1_val_"  +str(nentries)+".npy", X1_val)
            np.save(folder + global_name + "/w0_val_"  +str(nentries)+".npy", w0_val)
            np.save(folder + global_name + "/w1_val_"  +str(nentries)+".npy", w1_val)
            np.save(folder + global_name + "/X0_train_"+str(nentries)+".npy", X0_train)
            np.save(folder + global_name + "/X1_train_"+str(nentries)+".npy", X1_train)
            np.save(folder + global_name + "/w0_train_"  +str(nentries)+".npy", w0_train)
            np.save(folder + global_name + "/w1_train_"  +str(nentries)+".npy", w1_train)
            if large_weight_clipping or weight_preprocess:
                np.save(folder + global_name + "/w0_train_raw_"  +str(nentries)+".npy", raw_w0_train)
                np.save(folder + global_name + "/w1_train_raw_"  +str(nentries)+".npy", raw_w1_train)
                np.save(folder + global_name + "/w0_val_raw_"  +str(nentries)+".npy", raw_w0_val)
                np.save(folder + global_name + "/w1_val_raw_"  +str(nentries)+".npy", raw_w1_val)
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
        plot_resampledRatio=False,
        ext_binning = None,
        ext_plot_path=None,
        verbose=False,
        normalise = False,
        scaling="minmax",
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
            logger.info("Extracting numpy data features")

        # Get data - only needed for column names which we can use features instead
        #x0df, weights0, labels0 = load(f = pathA,
        #                     features=features, weightFeature=weightFeature,
        #                     n = int(nentries), t = TreeName)
        #x1df, weights1, labels1 = load(f = pathB,
        #                     features=features, weightFeature=weightFeature,
        #                     n = int(nentries), t = TreeName)

        # load samples
        X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0,  name="nominal features")
        X0 = np.nan_to_num(X0, nan=-1.0, posinf = 0.0, neginf=0.0)
        X1 = load_and_check(x1, memmap_files_larger_than_gb=1.0, name="variation features")
        X1 = np.nan_to_num(X1, nan=-1.0, posinf = 0.0, neginf=0.0)
        W0 = load_and_check(w0, memmap_files_larger_than_gb=1.0, name="nominal weights")
        W1 = load_and_check(w1, memmap_files_larger_than_gb=1.0, name="variation weights")

        if isinstance(metaData, str):
            metaDataFile = open(metaData, 'rb')
            metaDataDict = pickle.load(metaDataFile)
            metaDataFile.close()
        else:
            metaDataDict = metaData
        #weights = weights / weights.sum() * len(X1)

        # Calculate the maximum of each column and minimum and then allocate bins
        if verbose:
            logger.info("Calculating min/max range for plots & binning")
        binning = defaultdict()
        minmax = defaultdict()
        divisions = 100 # 50 default

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
            # Check for integer values in plotting data only, this indicates that no capping on data range needed
            #  as integer values indicate well bounded data
            intTest = [ (i % 1) == 0  for i in X0[:,idx] ]
            intTest = all(intTest) #np.all(intTest == True)
            upperThreshold = 100 if intTest or np.any(X0[:,idx] < 0) else 98
            max = np.percentile(X0[:,idx], upperThreshold)
            lowerThreshold = 0 if (np.any(X0[:,idx] < 0 ) or intTest) else 0
            min = np.percentile(X0[:,idx], lowerThreshold)
            minmax[idx] = [min,max]
            binning[idx] = np.linspace(min, max, divisions)
            if verbose:
                logger.info("<loading.py::load_result>::   Column {}:  min  =  {},  max  =  {}".format(column,min,max))
                print(binning[idx])       

        # no point in plotting distributions with too few events, they only look bad
        #if int(nentries) > 5000:
        # plot ROC curves
        logger.info("<loading.py::load_result>::   Printing ROC")
        if plot_ROC:
            draw_ROC(X0, X1, W0, W1, weights, label, global_name, nentries, plot)
        if plot_obs_ROC:
            draw_Obs_ROC(X0, X1, W0, W1, weights, label, global_name, nentries, plot, plot_resampledRatio)

        if verbose:
            logger.info("<loading.py::load_result>::   Printing weighted distributions")
        # plot reweighted distributions
        draw_weighted_distributions(
            X0, X1, W0, W1,
            weights,
            metaDataDict.keys(),#x0df.columns,
            binning,
            label,
            global_name, nentries, plot, ext_plot_path,
            normalise,
        )

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
