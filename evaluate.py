import os
import sys
import logging
import numpy as np
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
    spectator_file_map = {"x0": "x0", "x1": "x1", "w0": "w0", "w1": "w1"}

    # setting up default data files
    data_files_path = {
        "model": f"{data_path}/models/{tag_name}_carl_{nevent}",
        "train": {
            x: f"{data_path}/data/{tag_name}/{y}_train_{nevent}.npy"
            for x, y in features_file_map.items()
        }.update({"metaData": metadata}),
        "val": {
            x: f"{data_path}/data/{tag_name}/{y}_val_{nevent}.npy"
            for x, y in features_file_map.items()
        }.update({"metaData": metadata}),
        "spectator-train": {
            x: f"{data_path}/data/{tag_name}/{y}_train_{nevent}.npy"
            for x, y in spectator_file_map.items()
        }.update({"metaData": spec_meta}),
        "spectator-val": {
            x: f"{data_path}/data/{tag_name}/{y}_val_{nevent}.npy"
            for x, y in spectator_file_map.items()
        }.update({"metaData": spec_meta}),
    }
    # check user specified data files
    if user_config is not None:
        with open(user_config, "r") as f:
            data_files_path.update(json.load(f))
        logger.info(f"updated data file: {data_files_path}")

    # check trained files
    check_lists = [
        data_files_path["train"],
        data_files_path["val"],
        data_files_path["spectator-train"],
        data_files_path["spectator-val"],
    ]
    for check_list in check_lists:
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
carl.load(data_files["model"])

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

    # Weight clipping if requested by user
    if carl_weight_clipping:
        carl_w_clipping = np.percentile(w, carl_weight_clipping)
        w[w > carl_w_clipping] = carl_w_clipping

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
