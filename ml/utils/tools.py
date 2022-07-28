from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
import os

# import stat
import numpy as np
import uproot
import pandas as pd

# import torch
# from torch.nn import functional as F
from collections import defaultdict

# from contextlib import contextmanager
# import pickle
from time import process_time  # For sub-process timing

# import cudf
# import torch

logger = logging.getLogger(__name__)

initialized = False


def GenerateFractionSamples(x, w, frac=0.50):
    """
    Randomly samples fraction of the events from given features and weight.

    Args:
        x : panda.DataFrame
            dataframe that contains the features for training.

        w : panda.DataFrame
            dataframe that contains the MC event weight.

        frac : float, optional
            fraction of samples to be re-sample from the original dataframe

    """

    frac_x = x.sample(frac=frac, random_state=42)
    frac_w = w.iloc[frac_x.index]

    return frac_x, frac_w


def AddInvertWeight(x, w, frac_x, frac_w):
    """
    append frac_x and frac_w them the original x and w data.
    Inverted the sign of frac_w to get inv_frac_w, and then append them into the
    orignal data set as well. The final distribution. The overall distribution
    shouldn't change since the frac_w and inv_frac_w cancel each other.

    Args:
        x : panda.DataFrame
            dataframe that contains the features for training.

        w : panda.DataFrame
            dataframe that contains the MC event weight.

        frac : float, optional
            fraction of samples to be re-sample from the original dataframe

    """
    # appending this into the original data frame
    x = x.append(frac_x)
    w = w.append(frac_w)

    # inverting the sign of weight, and adding it to the data frame
    frac_w *= -1
    if "polarity" in frac_x:
        frac_x["polarity"] *= -1
    x = x.append(frac_x)
    w = w.append(frac_w)

    """
    with open("addInvSample.pkl", "wb") as f:
        frac_x = frac_x[sorted(frac_x.columns)]
        addInvSample = (frac_x, frac_w)
        pickle.dump(addInvSample, f)
    """

    return x, w


def HarmonisedLoading(
    fA="",
    fB="",
    features=[],
    spectators=[],
    weightFeature="DummyEvtWeight",
    nentries=-1,
    TreeName="Tree",
    Filter=None,
    do_self_dope=False,
    do_mix_dope=False,
    weight_polarity=False,
):
    """
    Harmonising feature and weight dataframe to same shape. i.e jet multiplicity
    for each event can be different in nominal and variational samples, which
    require matching to the minimum jet multiplicity.

    Args:
        fA: str
            file name to (nominal) N-tuples

        fB: str
            file name to (variational) N-tuples

        features: list(str)
            list of features (branch name) in the N-tuple TTree.

        spectators: list(str)
            list of features that will not be in the training.
            Only serves as spectator to the trained model.

        weightFeasure: str, default="DummyEvtWeight"
            name of the weigth branch in TTree.

        nentries: int
            number of events for training.

        TreeName: str
            Name of the TTree.

        do_self_dope: bool, default=False
            Take fraction of the samples from fA and invert the sign of the
            event weight, then append both the fraction and inverted fraction to
            the orignal sample set.

        do_mix_dope: bool, default=False
            similar to do_self_dope, but using fB sample set to generate the
            fraction of events instead.

        weight_polarity: bool, default=False
            adding a polarity feature based on the sign of the event weight.
            i.e. polarity=1 (-1) will be assigned to positive(negative) event weignt.
    Return:
        x0: pandas.DataFrame
            dataframe contains features from sample-0

        spec_x0: pandas.DataFrame or None
            dataframe contains spectators from sample-0

        w0: pandas.DataFrame
            the corresponding (event) weight from sample-0

        vlabels0: list(str)
            list of labels for features from samples-0

        x1: pandas.DataFrame
            dataframe contains features from sample-1

        spec_x1: pandas.DataFrame or None
            dataframe contains spectators from sample-1

        w1: pandas.DataFrame
            the corresponding (event) weight from sample-1

        vlabels1: list(str)
            list of labels for features from samples-1
    """

    x0, spec_x0, w0, vlabels0 = load(
        f=fA,
        features=features,
        spectators=spectators,
        weightFeature=weightFeature,
        n=int(nentries),
        t=TreeName,
        Filter=Filter,
        weight_polarity=weight_polarity,
    )

    x1, spec_x1, w1, vlabels1 = load(
        f=fB,
        features=features,
        spectators=spectators,
        weightFeature=weightFeature,
        n=int(nentries),
        t=TreeName,
        Filter=Filter,
        weight_polarity=weight_polarity,
    )

    x0, x1 = CoherentFlattening(x0, x1)
    spec_x0 = flatten_nested_column(spec_x0)
    spec_x1 = flatten_nested_column(spec_x1)

    # hard-coding the event weight inverting here for the moment
    # TODO: make this optional
    if do_self_dope:
        frac_x0, frac_w0 = GenerateFractionSamples(x0, w0)
        x0, w0 = AddInvertWeight(x0, w0, frac_x0, frac_x0)
    if do_mix_dope:
        frac_x1, frac_w1 = GenerateFractionSamples(x1, w1)
        x0, w0 = AddInvertWeight(x0, w0, frac_x1, frac_x1)

    # removing Nans and Inf
    x0.fillna(0, inplace=True)
    w0.fillna(0, inplace=True)
    x1.fillna(0, inplace=True)
    w1.fillna(0, inplace=True)

    return x0, w0, vlabels0, spec_x0, x1, w1, vlabels1, spec_x1


def CoherentFlattening(df0, df1):

    # Find the lowest common denominator for object lengths
    df0_objects = df0.select_dtypes(object)
    # df1_objects = df1.select_dtypes(object) # this one never used?
    minObjectLen = defaultdict()
    # Find the columns that are not scalars and get all object type columns
    # maxListLength = df.select_dtypes(object).apply(lambda x: x.list.len()).max(axis=1)
    print(df0)
    for column in df0_objects:
        elemLen0 = df0[column].apply(lambda x: len(x)).max()
        elemLen1 = df1[column].apply(lambda x: len(x)).max()

        # Warn user
        if elemLen0 != elemLen1:
            print(
                "<tools.py::CoherentFlattening()>::   The two datasets do not have the same length for features '{}', please be warned that we choose zero-padding using lowest dimensionatlity".format(
                    column
                )
            )

        minObjectLen[column] = elemLen0 if elemLen0 < elemLen1 else elemLen1
        print(
            "<tools.py::CoherentFlattening()>::   Variable: {}({}),   min size = {}".format(
                column, df0[column].dtypes, minObjectLen
            )
        )
        print(
            "<tools.py::CoherentFlattening()>::      Element Length 0 = {}".format(
                elemLen0
            )
        )
        print(
            "<tools.py::CoherentFlattening()>::      Element Length 1 = {}".format(
                elemLen1
            )
        )

        # Now break up each column into elements of max size 'macObjectLen'
        # df0_flattened = pd.DataFrame(df0[column].to_list(), columns=[column+str(idx) for idx in range(elemLen0)])
        df0_flattened = pd.DataFrame(
            df0.pop(column).to_list(),
            columns=[column + str(idx) for idx in range(elemLen0)],
        )
        # print(df0_flattened)
        # df0_flattened = df0_flattened.fillna(0)
        # print(df0_flattened)

        # Delete extra dimensions if needed due to non-matching dimensionality of df0 & df1
        if elemLen0 > minObjectLen[column]:
            delColumns0 = [
                column + str(idx) for idx in range(minObjectLen[column], elemLen0)
            ]
            print(
                "<tools.py::CoherentFlattening()>::   Deleting {}".format(delColumns0)
            )
            for idx in range(minObjectLen[column], elemLen0):
                del df0_flattened[column + str(idx)]

        # print(df[column])
        # print(df_flattened)
        # del df0[column]
        df0 = df0.join(df0_flattened)
        print(df0)

        # Now break up each column into elements of max size 'macObjectLen'
        # df1_flattened = pd.DataFrame(df1[column].to_list(), columns=[column+str(idx) for idx in range(elemLen1)])
        df1_flattened = pd.DataFrame(
            df1.pop(column).to_list(),
            columns=[column + str(idx) for idx in range(elemLen1)],
        )
        # df1_flattened = df1_flattened.fillna(0)

        # Delete extra dimensions if needed due to non-matching dimensionality of df0 & df1
        if elemLen1 > minObjectLen[column]:
            delColumns1 = [
                column + str(idx) for idx in range(minObjectLen[column], elemLen1)
            ]
            print(
                "<tools.py::CoherentFlattening()>::   Deleting {}".format(delColumns1)
            )
            for idx in range(minObjectLen[column], elemLen1):
                del df1_flattened[column + str(idx)]

        # print(df[column])
        # print(df_flattened)
        # del df1[column]
        df1 = df1.join(df1_flattened)
        print(df1)

    print("<loading.py::load()>::    Flattened Dataframe")
    # print(df)
    return df0, df1


def flatten_nested_column(df):
    """
    flatten nested columns.
    """
    if df is not None:
        df_objects = df.select_dtypes(object)
        for column in df_objects:
            maxLen = df[column].apply(lambda x: len(x)).max()
            new_col = [column + str(idx) for idx in range(maxLen)]
            df = df.join(pd.DataFrame(df.pop(column).to_list(), columns=new_col))
            df.fillna(0, inplace=True)
    return df


def load(
    f="",
    features=[],
    spectators=[],
    weightFeature="DummyEvtWeight",
    n=-1,
    t="Tree",
    weight_polarity=False,
    Filter=None,
):
    """
    Function for preparing feastures for training.

    Args:
        f : str
            Path to the ROOT N tuples.

        features : [], optional
            List of observables/features name inside the ROOT file TTree.
            If no feature is provided, all branches from the TTree will be used.

        spectators: [], optional
            List of observables/features name inside the ROOT file TTree.
            These features will NOT be part of the training, and only serve as
            spectator for evaluating trained model performance.

        weightFeature : str, optional
            Name of the branch that contains the MC event weight.
            If no weightFeasure is provided, weight of 1 will be used.

        n : int, optional
            Total number of input events.

        weight_polarity : boolean, optional
            introduce a polarity feature for training. The value of the polarity
            will be determined by the sign of the event weight. 1 will be assigned
            to positive (>=0) event weight, and -1 will be assigned to negative (< 0)
            event weight.

        Filter : instance of Filter.Filter, optional
            An instance of Filter.Filter that contains (numpy) expression that
            can be evaluated by panda.dataframe.eval.
            (Essentially the expressing is evaluete by panda.eval(), with
            default backend `numexpr`)

    Return:
        ( pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, list(str) ) :

            DataFrame of feasures, spectators, event weight, and labels
    """
    # grab our data and iterate over chunks of it with uproot
    logger.info(f"<{process_time()}> Uproot open file")
    file = uproot.open(f)

    # Now get the Tree
    logger.info(f"<{process_time()}> Getting TTree from file")
    X_tree = file[t]

    # Check that features were set by user, if not then will use all features
    #   -  may double up on weight feature but will warn user
    if not features:
        # Set the features to all keys in tree - warn user!!!
        logger.info(
            "Attempting extract features however user did not define values. Using all keys inside TTree as features."
        )
        features = X_tree.keys()

    # Ensure that we don't try to load 0 events. Convert 0 to None, meaning load the entire dataset
    if n == 0:
        n = None

    # Extract the pandas dataframe - warning about jagged arrays
    logger.info(f"<{process_time()}> Converting uproot array to panda's dataframe")
    # load both features and spector into the same dataframe
    df = pd.DataFrame(X_tree.arrays(features + spectators, library="np", entry_stop=n))

    # Implement GPU capable dataframe loading/caching, as we want to speed up data processing - needs a docker image for library "cp"
    # if torch.cuda.is_available() and False:
    #    df = cudf.DataFrame(X_tree.arrays(features, library="cp", entry_stop=n))
    # else:
    #    df = pd.DataFrame(X_tree.arrays(features, library="np", entry_stop=n))

    # Extract the weights from the Tree if specificed
    logger.info(f"<{process_time()}> Obtaining data point weights (dataframe)")
    if weightFeature == "DummyEvtWeight":
        dweights = np.ones(len(df.index))
        weights = pd.DataFrame(
            data=dweights, index=range(len(df.index)), columns=[weightFeature]
        )
    else:
        weights = pd.DataFrame(X_tree.arrays(weightFeature, library="np", entry_stop=n))

    # Apply filtering if set by user
    if Filter != None:
        logger.info(f"<{process_time()}> Applying filtering")
        for logExp in Filter.FilterList:
            if all([var in df for var in Filter.variables]):
                df_mask = df.eval(logExp)
            else:
                # if the data frame does not contains all of the variables in
                # the filter `logExp`, it will throw a KeyError.
                # We then need to contruct a temperoray dataframe with all variables
                # from `logExp` in order to compute the mask.
                df_filter = pd.DataFrame(
                    X_tree.arrays(Filter.variables, library="np", entry_stop=n)
                )
                df_mask = df_filter.eval(logExp)
            df = df[df_mask]
            weights = weights[df_mask]

    # Reset all row numbers
    logger.info(
        f"<{process_time()}> Re-setting row numbers in panda dataframes due to filtering or shuffling of dataset"
    )
    df = df.reset_index(drop=True)
    weights = weights.reset_index(drop=True)

    # For the moment one should simply use the features
    if weight_polarity:
        logger.info(
            f"<{process_time()}> Converting all weights to positive and adding weight polarity as new additonal training feature"
        )
        polarity_name = "polarity"
        df[polarity_name] = weights[weightFeature].apply(lambda x: 1 if x >= 0 else -1)
        labels = features + [polarity_name]
    else:
        labels = features

    # split spectator into separate dataframe
    if spectators:
        spectator_df = pd.concat([df.pop(x) for x in spectators], axis=1)
    else:
        spectator_df = None

    logger.info(f"<{process_time()}> Completed dataframe loading")
    return (df, spectator_df, weights, labels)


def create_missing_folders(folders):
    if folders is None:
        return

    for folder in folders:
        if folder is None or folder == "":
            continue

        if not os.path.exists(folder):
            os.makedirs(folder)

        elif not os.path.isdir(folder):
            raise OSError("Path {} exists, but is no directory!".format(folder))


def load_and_check(
    filename, warning_threshold=1.0e9, memmap_files_larger_than_gb=None, name=None
):
    if filename is None:
        return None

    # in case filenmae is not string, the warning does not show which file it is
    # so it will tell you where the large numbers are coming from instead of
    # showing just a list of values from the arrays.
    name = name or filename

    if not isinstance(filename, six.string_types):
        data = filename
        memmap = False
    else:
        filesize_gb = os.stat(filename).st_size / 1.0 * 1024**3
        if (
            memmap_files_larger_than_gb is None
            or filesize_gb <= memmap_files_larger_than_gb
        ):
            logger.info(f"Loading {filename} into RAM")
            data = np.load(filename)
            memmap = False
        else:
            logger.info(f"Loading {filename} as memory map")
            data = np.load(filename, mmap_mode="c")
            memmap = True

    if not memmap:
        n_nans = np.sum(np.isnan(data))
        n_infs = np.sum(np.isinf(data))
        n_finite = np.sum(np.isfinite(data))
        if n_nans + n_infs > 0:
            logger.warning(
                f"{name} contains {n_nans} NaNs and {n_infs} Infs, compared to {n_finite} finite numbers!"
            )

        smallest = np.nanmin(data)
        largest = np.nanmax(data)
        if np.abs(smallest) > warning_threshold or np.abs(largest) > warning_threshold:
            logger.warning(
                f"Warning: file {name} has some large numbers, ranging from {smallest} to {largest}"
            )

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    return data


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
