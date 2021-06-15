from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
import os
import stat
import numpy as np
import uproot
import pandas as pd
import torch
from torch.nn import functional as F
from collections import defaultdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

initialized = False

def HarmonisedLoading(fA="",
                      fB="",
                      features=[],
                      weightFeature="DummyEvtWeight",
                      nentries=0,
                      TreeName="Tree"
                  ):


    x0, w0, vlabels0 = load(f = fA,
                            features=features, weightFeature=weightFeature,
                            n = int(nentries), t = TreeName)
    x1, w1, vlabels1 = load(f = fB,
                            features=features, weightFeature=weightFeature,
                            n = int(nentries), t = TreeName)

    x0, x1 = CoherentFlattening(x0,x1)

    return x0, w0, vlabels0, x1, w1, vlabels1



def CoherentFlattening(df0, df1):

    # Find the lowest common denominator for object lengths
    df0_objects = df0.select_dtypes(object)
    df1_objects = df1.select_dtypes(object)
    minObjectLen = defaultdict()
    for column in df0_objects:
        elemLen0 = df0[column].apply(lambda x: len(x)).max()
        elemLen1 = df1[column].apply(lambda x: len(x)).max()

        # Warn user
        if elemLen0 != elemLen1:
            print("<tools.py::CoherentFlattening()>::   The two datasets do not have the same length for features '{}', please be warned that we choose zero-padding using lowest dimensionatlity".format(column))

        minObjectLen[column] = elemLen0 if elemLen0 < elemLen1 else elemLen1
        print("<tools.py::CoherentFlattening()>::   Variable: {}({}),   min size = {}".format( column, df0[column].dtypes, minObjectLen))
        print("<tools.py::CoherentFlattening()>::      Element Length 0 = {}".format( elemLen0))
        print("<tools.py::CoherentFlattening()>::      Element Length 1 = {}".format( elemLen1))

    # Find the columns that are not scalars and get all object type columns
    #maxListLength = df.select_dtypes(object).apply(lambda x: x.list.len()).max(axis=1)
    for column in df0_objects:
        elemLen0 = df0[column].apply(lambda x: len(x)).max()
        elemLen1 = df1[column].apply(lambda x: len(x)).max()


        # Now break up each column into elements of max size 'macObjectLen'
        df0_flattened = pd.DataFrame(df0[column].to_list(), columns=[column+str(idx) for idx in range(elemLen0)])
        df0_flattened = df0_flattened.fillna(0)

        # Delete extra dimensions if needed due to non-matching dimensionality of df0 & df1
        if elemLen0 > minObjectLen[column]:
            delColumns0 = [column+str(idx) for idx in range(minObjectLen[column], elemLen0)]
            print("<tools.py::CoherentFlattening()>::   Deleting {}".format(delColumns0))
            for idx in range(minObjectLen[column], elemLen0):
                del df0_flattened[column+str(idx)]

        #print(df[column])
        #print(df_flattened)
        del df0[column]
        df0 = df0.join(df0_flattened)

        # Now break up each column into elements of max size 'macObjectLen'
        df1_flattened = pd.DataFrame(df1[column].to_list(), columns=[column+str(idx) for idx in range(elemLen1)])
        df1_flattened = df1_flattened.fillna(0)

        # Delete extra dimensions if needed due to non-matching dimensionality of df0 & df1
        if elemLen1 > minObjectLen[column]:
            delColumns1 = [column+str(idx) for idx in range(minObjectLen[column], elemLen1)]
            print("<tools.py::CoherentFlattening()>::   Deleting {}".format(delColumns1))
            for idx in range(minObjectLen[column], elemLen1):
                del df1_flattened[column+str(idx) ]

        #print(df[column])
        #print(df_flattened)
        del df1[column]
        df1 = df1.join(df1_flattened)

    print("<loading.py::load()>::    Flattened Dataframe")
    #print(df)
    return df0,df1

def load(
    f="",
    features=[],
    weightFeature="DummyEvtWeight",
    n=0,
    t="Tree"
):
    # grab our data and iterate over chunks of it with uproot
    print("Uproot open file")
    file = uproot.open(f)

    # Now get the Tree
    print("Getting TTree from file")
    X_tree = file[t]

    # Check that features were set by user, if not then will use all features
    #   -  may double up on weight feature but will warn user
    if not features:
        # Set the features to all keys in tree - warn user!!!
        print("<tools.py::load()>::   Attempting extract features however user did not define values. Using all keys inside TTree as features.")
        features = X_Tree.keys()

    # Extract the pandas dataframe - warning about jagged arrays
    #df = X_tree.pandas.df(features, flatten=False)
    df = pd.DataFrame(X_tree.arrays(features, library="np", entry_stop=n))

    # Extract the weights from the Tree if specificed
    if weightFeature == "DummyEvtWeight":
        #weights = len(df.index)
        dweights = np.ones(len(df.index))
        weights = pd.DataFrame(data=dweights, index=range(len(df.index)), columns=[weightFeature])
    else:
        #weights = X_tree[weightFeature]
        #weights = X_tree.pandas.df(weightFeature)
        weights = pd.DataFrame(X_tree.arrays(weightFeature, library="np", entry_stop=n))

    # For the moment one should siply use the features
    labels  = features

    return (df, weights, labels)


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


def load_and_check(filename, warning_threshold=1.0e9, memmap_files_larger_than_gb=None):
    if filename is None:
        return None

    if not isinstance(filename, six.string_types):
        data = filename
        memmap = False
    else:
        filesize_gb = os.stat(filename).st_size / 1.0 * 1024 ** 3
        if memmap_files_larger_than_gb is None or filesize_gb <= memmap_files_larger_than_gb:
            logger.info("  Loading %s into RAM", filename)
            data = np.load(filename)
            memmap = False
        else:
            logger.info("  Loading %s as memory map", filename)
            data = np.load(filename, mmap_mode="c")
            memmap = True

    if not memmap:
        n_nans = np.sum(np.isnan(data))
        n_infs = np.sum(np.isinf(data))
        n_finite = np.sum(np.isfinite(data))
        if n_nans + n_infs > 0:
            logger.warning(
                "%s contains %s NaNs and %s Infs, compared to %s finite numbers!", filename, n_nans, n_infs, n_finite
            )

        smallest = np.nanmin(data)
        largest = np.nanmax(data)
        if np.abs(smallest) > warning_threshold or np.abs(largest) > warning_threshold:
            logger.warning("Warning: file %s has some large numbers, rangin from %s to %s", filename, smallest, largest)

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
