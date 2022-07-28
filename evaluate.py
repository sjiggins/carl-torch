import logging
import numpy as np
from ml import RatioEstimator
from ml.utils.loading import Loader
from arg_handler import arg_handler_eval
from file_path_handler import datafiles_path_preparation

logger = logging.getLogger(__name__)

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
