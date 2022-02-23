import os
import sys
import logging
import tarfile
import pickle
import pathlib
import numpy as np
from itertools import repeat

from ml import RatioEstimator
from ml import Loader
from ml import Filter
from arg_handler import arg_handler_train

logger = logging.getLogger(__name__)

#################################################
# Arugment parsing
opts = arg_handler_train()
nominal  = opts.nominal
variation = opts.variation
n = opts.nentries
p = opts.datapath
global_name = opts.global_name
features = opts.features.split(",")
spectators = opts.spectators.split(",") if opts.spectators else []
weightFeature = opts.weightFeature
treename = opts.treename
binning = opts.binning
n_hidden = tuple(opts.layers) if opts.layers != None else tuple( repeat( (len(features)), 3) )
batch_size = opts.batch_size
per_epoch_plot = opts.per_epoch_plot
per_epoch_save = opts.per_epoch_save
per_epoch_stats = opts.per_epoch_stats
nepoch = opts.nepoch
scale_method = opts.scale_method
weight_clipping = opts.weight_clipping
weight_sigma = opts.weight_nsigma
polarity = opts.polarity
loss_type = opts.loss_type
BoolFilter = opts.BoolFilter
#################################################

#################################################
if os.path.exists(f"data/{global_name}/data_out.tar.gz"):
    # tar = tarfile.open("data_out.tar.gz", "r:gz")
    tar = tarfile.open(f"data/{global_name}/data_out.tar.gz")
    tar.extractall()
    tar.close()

logger.info(f"Trying to do training of model with datasets: {nominal=}, {variation=}, {n} events")
logger.info("Checkfing input files")
# path to ROOT Ntuples files and list of preprocessed numpy arrays
input_ntuple_files = [
    f"{p}/{nominal}.root",
    f"{p}/{variation}.root",
]
prepared_data = {
    "X_train" : f"data/{global_name}/X_train_{n}.npy",
    "y_train" : f"data/{global_name}/y_train_{n}.npy",
    "w_train" : f"data/{global_name}/w_train_{n}.npy",
    "X0_train" : f"data/{global_name}/X0_train_{n}.npy",
    "w0_train" : f"data/{global_name}/w0_train_{n}.npy",
    "X1_train" : f"data/{global_name}/X1_train_{n}.npy",
    "w1_train" : f"data/{global_name}/w1_train_{n}.npy",
    "X0_val" : f"data/{global_name}/X0_val_{n}.npy",
    "w0_val" : f"data/{global_name}/w0_val_{n}.npy",
    "X1_val" : f"data/{global_name}/X1_val_{n}.npy",
    "w1_val" : f"data/{global_name}/w1_val_{n}.npy",
}
file_paths = list(prepared_data.values())
file_paths.append(f"data/{global_name}/metaData_{n}.pkl")
# Check if already pre-processed numpy arrays exist
if all(map(os.path.exists, file_paths)):
    prepared_data["spectators"] = list(set(spectators)-set(features))
    with open(file_paths[-1], "rb") as f:
        _metadata = pickle.load(f)
        prepared_data["metaData"] = _metadata
        prepared_data["features"] = list(_metadata.keys())
else:
    # if missing any of the pre-processed files
    # try to check Ntuple files and run loading
    if not all(map(os.path.exists, input_ntuple_files)):
        logger.warning(f"Unable to find N-tuple files {input_ntuple_files}")
        sys.exit()

    # prepare loading of data from root of numpy arrays
    loading = Loader()
    if BoolFilter != None:
        InputFilter = Filter(FilterString = BoolFilter)
        loading.Filter= InputFilter

    prepared_data = loading.loading(
        folder=f"{pathlib.Path('./data/').resolve()}/",
        plot=True,
        global_name=global_name,
        features=features,
        spectators=spectators,
        weightFeature=weightFeature,
        TreeName=treename,
        randomize=False,
        save=True,
        correlation=True,
        preprocessing=False,
        nentries=n,
        pathA=f"{p}/{nominal}.root",
        pathB=f"{p}/{variation}.root",
        noTar=True,
        normalise=False,
        debug=False,
        weight_preprocess=weight_sigma > 0,
        weight_preprocess_nsigma=weight_sigma,
        large_weight_clipping=weight_clipping,
        weight_polarity=polarity,
        scaling=scale_method,
    )
    logger.info(" Loaded new datasets ")

#######################################

#######################################
# Estimate the likelihood ratio using a NN model
#   -> Calculate number of input variables as rudimentary guess
structure = n_hidden
# Use the number of inputs as input to the hidden layer structure
estimator = RatioEstimator(
    n_hidden=(structure),
    activation="relu",
)
estimator.scaling_method = scale_method
if opts.dropout_prob is not None:
    estimator.dropout_prob = opts.dropout_prob

# per epoch plotting
if per_epoch_plot:
    plots_args = {
        "metaData":prepared_data["metaData"],
        "features":features,
        "plot":True,
        "nentries":n,
        "global_name":global_name,
        "ext_binning":binning,
        "verbose" : False,
        "plot_ROC" : False,
        "plot_obs_ROC" : False,
        "normalise" : True, # plotting
    }
    prepared_data["per_epoch_plot"] = plots_args

# per epoch saving of model
if per_epoch_save:
    save_args = {
        "filename" : f"{global_name}_carl_{n}",
        "x" : prepared_data.get("X_train", None),
        "metaData" : prepared_data.get("metaData", None),
        "save_model" : True,
        "export_model" : True,
    }
    prepared_data["per_epoch_save"] = save_args

# additional options to pytorch training package
kwargs = {}
if opts.regularise is not None:
    logger.info("L2 loss regularisation included.")
    kwargs={"weight_decay": 1e-5}

# getting n_workers for DataLoader
try:
    n_workers = len(os.sched_getaffinity(0))
except Exception:
    n_workers = os.cpu_count()

if per_epoch_stats:
    stats_method_list = ["compute_kl_divergence", "wasserstein"]
else:
    stats_method_list = []

# perform training
train_loss, val_loss, accuracy_train, accuracy_val = estimator.train(
    method='carl',
    batch_size=batch_size,
    n_epochs=nepoch,
    validation_split=0.25,
    #optimizer="amsgrad",
    input_data_dict=prepared_data,
    scale_inputs=True,
    early_stopping=False,
    #early_stopping_patience=20,
    intermediate_train_plot = per_epoch_plot,
    intermediate_save = per_epoch_save,
    intermediate_stats_dist = per_epoch_stats,
    stats_method_list = stats_method_list,
    optimizer_kwargs=kwargs,
    global_name=global_name,
    plot_inputs=True,
    nentries=n,
    loss_type=loss_type,
    n_workers=n_workers,
)

# saving loss values and final trained models
np.save(f"loss_train_{global_name}.npy", train_loss)
np.save(f"loss_val_{global_name}.npy", val_loss)
np.save(f"accuracy_train_{global_name}.npy", accuracy_train)
np.save(f"accuracy_val_{global_name}.npy", accuracy_val)
estimator.save(
    f"models/{global_name}_carl_{n}",
    prepared_data["X_train"],
    prepared_data["metaData"],
    export_model = True,
    noTar=True,
)
########################################
