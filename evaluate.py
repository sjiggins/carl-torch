import os
import sys
import logging
import numpy as np
from arg_handler import arg_handler_eval
import json
from ml import RatioEstimator
from ml.utils.loading import Loader
from arg_handler import arg_handler_eval

logger = logging.getLogger(__name__)


def datafiles_path_preparation(
    data_path, tag_name, nevent, user_config=None, skip_ok=True
):
    """
    Generate data files path for used in the evaluation.

    The user_config accepts file in JSON file format if specific file is request
    by the users.
    """

    metadata = f"{data_path}/data/{tag_name}/metaData_{nevent}.pkl"
    spec_meta = f"{data_path}/data/{tag_name}/metaData_{nevent}_spectator.pkl"

    # should hamonize the file naming in future
    features_file_map = {"x0": "X0", "x1": "X1", "w0": "w0", "w1": "w1"}
    spectator_file_map = {"x0": "spec_x0", "x1": "spec_x1", "w0": "w0", "w1": "w1"}

    # setting up default data files
    data_files_path = {
        "model": f"{data_path}/models/{tag_name}_carl_{nevent}",
        "train": {
            x: f"{data_path}/data/{tag_name}/{y}_train_{nevent}.npy"
            for x, y in features_file_map.items()
        },
        "val": {
            x: f"{data_path}/data/{tag_name}/{y}_val_{nevent}.npy"
            for x, y in features_file_map.items()
        },
        "spectator-train": {
            x: f"{data_path}/data/{tag_name}/{y}_train_{nevent}.npy"
            for x, y in spectator_file_map.items()
        },
        "spectator-val": {
            x: f"{data_path}/data/{tag_name}/{y}_val_{nevent}.npy"
            for x, y in spectator_file_map.items()
        },
    }
    # add meta data path
    for key in ["train", "val"]:
        data_files_path[key].update({"metaData": metadata})
    for key in ["spectator-train", "spectator-val"]:
        data_files_path[key].update({"metaData": spec_meta})
    # check user specified data files
    if user_config is not None:
        with open(user_config, "r") as f:
            user_fdata = json.load(f)
            logger.info(f"user file data: {user_fdata}")
            data_files_path.update(user_fdata)
    logger.info(f"data file: {data_files_path}")

    # check trained files
    check_lists = {
        "train": data_files_path["train"],
        "val": data_files_path["val"],
        "spec-train": data_files_path["spectator-train"],
        "spec-val": data_files_path["spectator-val"],
    }
    for name, check_list in check_lists.items():
        logger.info(f"Checking files for {name}")
        flist = check_list.values()
        if not all(map(os.path.exists, flist)):
            logger.warning(f"Cannot locate all of the files {flist}")
        if not skip_ok:
            logger.info("ABORTING")
            sys.exit()
        # setting it to empty dict
        check_list = {}

    return data_files_path


#################################################
opts = arg_handler_eval()
# nominal = opts.nominal
# variation = opts.variation
n = opts.nentries
p = opts.datapath
global_name = opts.global_name
features = opts.features.split(",")
weightFeature = opts.weightFeature
treename = opts.treename
binning = opts.binning
normalise = opts.normalise
scale_method = opts.scale_method
output = opts.output
carl_weight_clipping = opts.carl_weight_clipping
datafile = opts.datafile
#################################################

data_files = datafiles_path_preparation(output, global_name, n, datafile)

# loading model
loading = Loader()
carl = RatioEstimator()
carl.scaling_method = scale_method
carl.load(data_files["model"], data_files["train"].get("metaData", None))

evaluate = ["train", "val"]
for eval_mode in evaluate:
    if not data_files[eval_mode]:
        logger.warning(f"check input file path, skipping {eval_mode}")
        continue
    logger.info(f"Running evaluation for {eval_mode}")
    r_hat, s_hat = carl.evaluate(x=data_files[eval_mode]["x0"])
    logger.info(f"{s_hat=}")
    logger.info(f"{r_hat=}")
    w = 1.0 / r_hat  # I thought r_hat = p_{1}(x) / p_{0}(x) ???
    # Correct nan's and inf's to 1.0 corrective weights as they are useless in this instance. Warning
    # to screen should already be printed
    # if carl_weight_protection:
    w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)

    if opts.weight_protection:
        w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)

    # Weight clipping if requested by user
    if opts.weight_threshold < 100:
        carl_w_clipping = np.percentile(w, opts.weight_threshold)

    logger.info(f"{w=}")
    logger.info(f"Loading Result for {eval_mode}")
    loading.load_result(
        **data_files[eval_mode],
        weights=w,
        features=features,
        # weightFeature=weightFeature,
        label=eval_mode,
        plot=True,
        nentries=n,
        global_name=global_name,
        plot_ROC=opts.plot_ROC,
        plot_obs_ROC=opts.plot_obs_ROC,
        ext_binning=binning,
        normalise=normalise,
        scaling=scale_method,
        plot_resampledRatio=opts.plot_resampledRatio,
    )

    # attempt to plot spectators
    if not data_files[f"spectator-{eval_mode}"]:
        continue
    try:
        loading.load_result(
            **data_files[f"spectator-{eval_mode}"],
            weights=w,
            features=features,
            label=f"spectator_{eval_mode}",
            plot=True,
            nentries=n,
            global_name=global_name,
            plot_ROC=False,
            plot_obs_ROC=False,
            ext_binning=binning,
            normalise=normalise,
        )
    except Exception as _error:
        logger.warning(f"Unable to plot spectator distributions due to: {_error}")

# Evaluate performance
logger.info("Evaluate Performance of Model")
carl.evaluate_performance(
    x=f"{output}/data/{global_name}/X_val_{n}.npy",
    y=f"{output}/data/{global_name}/y_val_{n}.npy",
)
