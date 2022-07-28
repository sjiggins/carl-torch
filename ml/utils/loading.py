from __future__ import absolute_import, division, print_function, unicode_literals
# import os
# import time
import logging
import tarfile
import torch
import pickle
import numpy as np
# import pandas as pd
import seaborn as sns
# from pandas.plotting import scatter_matrix
# import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from functools import partial
from collections import OrderedDict
from .tools import create_missing_folders, load, load_and_check, HarmonisedLoading
from .reweighting import binned_reweighting
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
        spectators=[],
        weightFeature="DummyEvtWeight",
        TreeName = "Tree",
        randomize = False,
        save = False,
        correlation = True,
        preprocessing = False,
        nentries = -1,
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
        bin_reweighting = None,
    ):
        """
        This medthod use for loading features from ROOT N-tuples files and
        preparing data for the training.

        Parameters
        ----------
        folder : str, default=None
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format).

        plot : bool, optional, default=False
            make validation plots

        global_name : str, optional, default="Test"
            Name of containing folder for executed training or evaluation run

        features: list, default=[]
            list of features for training.

        spectators: list, default=[]
            list of features that DOES NOT go into the training.
            Use for examining the performace of a trained model on non-trained features.

        weightFeature: str, default="DummyEvtWeight"
            name of the branch in the ROOT N-tuples used for the event weight.
            If default is used, all event weights are assume to be 1.0

        randomize : bool, optional, default=False
            Randomize training sample.

        save : bool, optional, default=False
            Save training ans test samples.

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

        # Create folders for storage
        create_missing_folders([folder+'/'+global_name])
        create_missing_folders(['plots'])

        # remove overlapped spectators in features.
        spectators = [ spec for spec in spectators if spec not in features ]

        # Extract the TTree data as pandas dataframes
        (
            x0, w0, vlabels0, spec_x0,
            x1, w1, vlabels1, spec_x1
        )  = HarmonisedLoading(
            fA = pathA,
            fB = pathB,
            features=features,
            spectators=spectators,
            weightFeature=weightFeature,
            nentries = int(nentries),
            TreeName = TreeName,
            weight_polarity=weight_polarity,
            Filter=self.Filter
        )

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

        # sort dataframes alphanumerically
        x0 = x0[sorted(x0.columns)]
        x1 = x1[sorted(x1.columns)]

        # get metadata, i.e. max, min, mean, std of all the variables in the dataframes
        #metaData = defaultdict()
        metaData = OrderedDict()
        if scaling == "standard":
            metaData = {v : (x0[v].mean() , x0[v].std() ) for v in  x0.columns }
            logger.info("Storing Z0 Standard scaling metadata: {}".format(metaData))
        elif scaling == "minmax":
            #metaData = {v : OrderedDict({x0[v].min() if x0[v].min() < x1[v].min() else x1[v].min(), x0[v].max() if x0[v].max() > x1[v].max() else x1[v].max() } for v in  x0.columns) }
            metaData = {v : (x0[v].min() if x0[v].min() < x1[v].min() else x1[v].min(), x0[v].max() if x0[v].max() > x1[v].max() else x1[v].max())  for v in  x0.columns }
            for obs, value in metaData.items():
                logger.info(f"Storing minmax scaling for {obs}: (min,max) = {value}")
        else:
            for v in x0.columns:
                metaData[v] = None

        # temp saving for later debug
        np.save(f"input_x0.npy", x0.to_numpy())
        np.save(f"input_x1.npy", x1.to_numpy())
        np.save(f"input_w0.npy", w0.to_numpy())
        np.save(f"input_w1.npy", w1.to_numpy())
        if spec_x0 is not None:
            np.save(f"input_spec_x0.npy", spec_x0.to_numpy())
        if spec_x1 is not None:
            np.save(f"input_spec_x1.npy", spec_x1.to_numpy())

        # use bin normalization is use, turn off sum weight normalise
        if bin_reweighting is not None:
            x0, w0, x1, w1 = binned_reweighting(x0, w0, x1, w1, bin_reweighting)
            # turn off normalise
            normalise = False

        # Create target labels
        y0 = np.zeros(x0.shape[0])
        y1 = np.ones(x1.shape[0])

        # get an array of indices
        indices_0 = np.arange(x0.shape[0])
        indices_1 = np.arange(x1.shape[0])

        # Convert features and weights to numpy
        x0 = x0.to_numpy()
        x1 = x1.to_numpy()
        w0 = w0.to_numpy()
        w1 = w1.to_numpy()
        if normalise:
            w0 = w0 / (w0.sum())
            w1 = w1 / (w1.sum())

        lookup_input_mixing_0 = {
            "X0": x0,
            "y0": y0,
            "w0": w0,
            "indices0": indices_0,
        }
        lookup_input_mixing_1 = {
            "X1": x1,
            "y1": y1,
            "w1": w1,
            "indices1": indices_1,
        }
        x0_lookup_names = []
        x1_lookup_names = []
        x0_input_dataset = []
        x1_input_dataset = []
        for _lable, _data in lookup_input_mixing_0.items():
            x0_lookup_names.append(f"{_lable}_train")
            x0_lookup_names.append(f"{_lable}_val")
            x0_input_dataset.append(_data)
        for _lable, _data in lookup_input_mixing_1.items():
            x1_lookup_names.append(f"{_lable}_train")
            x1_lookup_names.append(f"{_lable}_val")
            x1_input_dataset.append(_data)

        # check if spectators
        if spectators and all(x is not None for x in [spec_x0, spec_x1]):
            spectator_metaData = OrderedDict()
            for spec in spec_x0:
                spectator_metaData[spec] = {spec_x0[spec].min(), spec_x0[spec].max()}
            spec_x0 = spec_x0.to_numpy()
            spec_x1 = spec_x1.to_numpy()
            x0_lookup_names += ["spec_x0_train", "spec_x0_val"]
            x1_lookup_names += ["spec_x1_train", "spec_x1_val"]
            x0_input_dataset.append(spec_x0)
            x1_input_dataset.append(spec_x1)
        else:
            spectator_metaData = None
            spec_x0 = None
            spec_x1 = None

        # Train, test splitting of input dataset
        #X0_train, X0_val, y0_train, y0_val, w0_train, w0_val =  train_test_split(x0, y0, w0, test_size=0.50, random_state=42)
        #X1_train, X1_val, y1_train, y1_val, w1_train, w1_val =  train_test_split(x1, y1, w1, test_size=0.50, random_state=42)
        prepared_data = {}
        for _name, split_set in zip(x0_lookup_names, train_test_split(*x0_input_dataset, test_size=0.50, random_state=42)):
            prepared_data[_name] = split_set
        for _name, split_set in zip(x1_lookup_names, train_test_split(*x1_input_dataset, test_size=0.50, random_state=42)):
            prepared_data[_name] = split_set

        # after splitting, we no longer need x0,x1,w0 etc, just set them to None
        x0 = w0 = y0 = spec_x0 = None
        x1 = w1 = y1 = spec_x1 = None

        #cliping large weights, and replace it by 1.0
        raw_w0_train = None
        raw_w1_train = None
        raw_w0_val = None
        raw_w1_val = None
        if large_weight_clipping or weight_preprocess:
            raw_w0_train = copy.deepcopy(prepared_data["w0_train"])
            raw_w1_train = copy.deepcopy(prepared_data["w1_train"])
            raw_w0_val = copy.deepcopy(prepared_data["w0_val"])
            raw_w1_val = copy.deepcopy(prepared_data["w1_val"])
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
            w0_train_sum = prepared_data["w0_train"].sum()
            w1_train_sum = prepared_data["w1_train"].sum()
            w0_val_sum = prepared_data["w0_val"].sum()
            w1_val_sum = prepared_data["w1_val"].sum()
            w0_train_per_change = (w0_train_sum - raw_w0_train_sum)/raw_w0_train_sum
            w1_train_per_change = (w1_train_sum - raw_w1_train_sum)/raw_w1_train_sum
            w0_val_per_change = (w0_val_sum - raw_w0_val_sum)/raw_w0_val_sum
            w1_val_per_change = (w1_val_sum - raw_w1_val_sum)/raw_w1_val_sum
            logger.info(f"After large weight clipping, training sum w0={w0_train_sum}, {w0_train_per_change}")
            logger.info(f"After large weight clipping, training sum w1={w1_train_sum}, {w1_train_per_change}")
            logger.info(f"After large weight clipping, validation sum w0={w0_val_sum}, {w0_val_per_change}")
            logger.info(f"After large weight clipping, validation sum w1={w1_val_sum}, {w1_val_per_change}")

        if weight_preprocess:
            w0_train_mean = np.mean(prepared_data["w0_train"])
            w1_train_mean = np.mean(prepared_data["w1_train"])
            w0_val_mean = np.mean(prepared_data["w0_val"])
            w1_val_mean = np.mean(prepared_data["w1_val"])
            w0_train_std = np.std(prepared_data["w0_train"])
            w1_train_std = np.std(prepared_data["w1_train"])
            w0_val_std = np.std(prepared_data["w0_val"])
            w1_val_std = np.std(prepared_data["w1_val"])

            local_buffers = {
                "w0_train" : {"weight" : prepared_data["w0_train"], "mean" : w0_train_mean, "std" : w0_train_std},
                "w1_train" : {"weight" : prepared_data["w1_train"], "mean" : w1_train_mean, "std" : w1_train_std},
                "w0_val" : {"weight" : prepared_data["w0_val"], "mean" : w0_val_mean, "std" : w0_val_std},
                "w1_val" : {"weight" : prepared_data["w1_val"], "mean" : w1_val_mean, "std" : w1_val_std},
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
        prepared_data["X_train"] = np.vstack([prepared_data["X0_train"], prepared_data["X1_train"]])
        prepared_data["y_train"] = np.concatenate((prepared_data["y0_train"], prepared_data["y1_train"]), axis=None)
        prepared_data["w_train"] = np.concatenate((prepared_data["w0_train"], prepared_data["w1_train"]), axis=None)

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
            # dict for tracking items being saved
            saving_items = {}
            additional_items = {
                "X_val" : np.vstack([prepared_data["X0_val"], prepared_data["X1_val"]]),
                "y_val" : np.concatenate((prepared_data["y0_val"], prepared_data["y1_val"]), axis=None),
                "w_val" : np.concatenate((prepared_data["w0_val"], prepared_data["w1_val"]), axis=None),
            }
            saving_items.update(prepared_data)
            saving_items.update(additional_items)

            # record the list of names before saving
            saving_items_names = list(saving_items.keys())
            # use pop to iterate through
            for name in saving_items_names:
                m_data = saving_items.pop(name)
                # check nans and inf
                n_nans = np.sum(np.isnan(m_data))
                n_infs = np.sum(np.isinf(m_data))
                if n_nans or n_infs:
                    n_finite = np.sum(np.isfinite(m_data))
                    logger.warning(f"{name} contains {n_nans=}, {n_infs=}, {n_finite=}")
                np.save(f"{folder}/{global_name}/{name}_{nentries}.npy", m_data)

            if large_weight_clipping or weight_preprocess:
                raw_saving_items = {
                    "w0_train_raw" : raw_w0_train,
                    "w1_train_raw" : raw_w1_train,
                    "w0_val_raw" : raw_w0_val,
                    "w1_val_raw" : raw_w1_val,
                }
                for name in list(raw_saving_items.keys()):
                    np.save(f"{folder}/{global_name}/{name}_{nentries}.npy", raw_saving_items.pop(name))
                    np.save(folder + global_name + "/w0_train_raw_"  +str(nentries)+".npy", raw_w0_train)

            # saving metadata
            metadata_fname = f"{folder}/{global_name}/metaData_{nentries}"
            with open(f"{metadata_fname}.pkl", "wb") as f:
                pickle.dump(metaData, f)
            if spectator_metaData:
                with open(f"{metadata_fname}_spectator.pkl", "wb") as f:
                    pickle.dump(spectator_metaData, f)

            # add metadata into prepared_data
            prepared_data.update({"metaData":metaData, "spectator_metaData":spectator_metaData})

            # update prepared_data with lookup for features and spectators from metadata
            prepared_data.update({"features" : list(metaData.keys())} )
            if spectator_metaData:
                prepared_data.update({"spectators" : list(spectator_metaData.keys())} )

            #Tar data files if training is done on GPU
            if torch.cuda.is_available() and not noTar:
                plot = False #don't plot on GPU...
                tar = tarfile.open("data_out.tar.gz", "w:gz")
                tar_list = [f"{folder}/{global_name}/{_name}_{nentries}.py" for _name in saving_items_names]
                for name in tar_list:
                    tar.add(name)
                    with open(metadata_fname, "wb") as f:
                        pickle.dump(metaData, f)
                tar.close()

        return prepared_data


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
        do_comparison=True,
        scaling="minmax",
        x0_index_mask=None,
        x1_index_mask=None,
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
        X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0, name="nominal features")
        X1 = load_and_check(x1, memmap_files_larger_than_gb=1.0, name="variation features")
        W0 = load_and_check(w0, memmap_files_larger_than_gb=1.0, name="nominal weights")
        W1 = load_and_check(w1, memmap_files_larger_than_gb=1.0, name="variation weights")

        if x0_index_mask is not None:
            _mask = load_and_check(x0_index_mask, name="x0 index mask")
            _mask = _mask.ravel()
            X0 = X0[_mask]
            W0 = W0[_mask]
            if weights is not None:
                weights = weights[_mask]
            _mask = None
        if x1_index_mask is not None:
            _mask = load_and_check(x1_index_mask, name="x1 index mask")
            _mask = _mask.ravel()
            X1 = X1[_mask]
            W1 = W1[_mask]
            _mask = None

        X0 = np.nan_to_num(X0, nan=-1.0, posinf = 0.0, neginf=0.0)
        X1 = np.nan_to_num(X1, nan=-1.0, posinf = 0.0, neginf=0.0)

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
        binning = OrderedDict()
        minmax = OrderedDict()
        divisions = 100 # 50 default

        # external binning from yaml file.
        if ext_binning:
            with open(ext_binning, "r") as f:
                ext_binning = yaml.load(f, yaml.FullLoader)

        #for idx,column in enumerate(x0df.columns):
        for idx, key in enumerate(metaDataDict):

            # check to see if variable is in the yaml file.
            # if not, proceed to automatic binning
            if ext_binning is not None:
                try:
                    bin_ranges = ext_binning["binning"][key]
                    binning[idx] = np.arange(*bin_ranges)
                    continue
                except KeyError:
                    logger.debug(f"cannot {key} in ext binning")
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
            metaDataDict.keys() if metaDataDict else features,#x0df.columns,
            binning,
            label,
            global_name, nentries, plot, ext_plot_path,
            normalise,
            do_comparison,
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
