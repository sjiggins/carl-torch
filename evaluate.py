import os
import sys
import logging
import numpy as np

from .ml import RatioEstimator
from .ml.utils.loading import Loader
from .arg_handler import arg_handler_eval

logger = logging.getLogger(__name__)

#################################################
opts, args = arg_handler_eval()
nominal  = opts.nominal
variation = opts.variation
n = opts.nentries
p = opts.datapath
global_name = opts.global_name
features = opts.features.split(",")
weightFeature = opts.weightFeature
treename = opts.treename
model = opts.model
binning = opts.binning
normalise = opts.normalise
raw_weight = opts.raw_weight
scale_method = opts.scale_method
output = opts.output
carl_weight_clipping = opts.weight_clipping
#################################################


train_data = f'{output}/data/{global_name}/X_train_{n}.npy'
metadata = f'{output}/data/{global_name}/metaData_{n}.pkl'
if os.path.exists(train_data) and os.path.exists(metadata):
    logger.info(" Doing evaluation of model trained with datasets: [{}, {}], with {} events.".format(nominal, variation, n))
else:
    logger.info(f"No data set directory of the form {train_data}.")
    logger.info("No datasets available for evaluation of model trained with datasets: [{},{}] with {} events.".format(nominal, variation, n))
    logger.info("ABORTING")
    sys.exit()

loading = Loader()
carl = RatioEstimator()
carl.scaling_method = scale_method
if model:
    carl.load(model)
else:
    carl.load(f'{output}/models/'+global_name+'_carl_'+str(n))
evaluate = ['train','val']
raw_w = "raw_" if raw_weight else ""
for i in evaluate:
    logger.info("Running evaluation for {}".format(i))
    r_hat, s_hat = carl.evaluate(x=f'{output}/data/{global_name}/X0_{i}_{n}.npy')
    logger.info("s_hat = {}".format(s_hat))
    logger.info("r_hat = {}".format(r_hat))
    w = 1./r_hat   # I thought r_hat = p_{1}(x) / p_{0}(x) ???
    # Correct nan's and inf's to 1.0 corrective weights as they are useless in this instance. Warning
    # to screen should already be printed
    #if carl_weight_protection:
    w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)

    # Weight clipping if requested by user
    if carl_weight_clipping:
        carl_w_clipping = np.percentile(w, carl_weight_clipping)
        w[w > carl_w_clipping] = carl_w_clipping

    print("w = {}".format(w))
    print("<evaluate.py::__init__>::   Loading Result for {}".format(i))
    loading.load_result(
        x0=f'{output}/data/{global_name}/X0_{i}_{n}.npy',
        x1=f'{output}/data/{global_name}/X1_{i}_{n}.npy',
        w0=f'{output}/data/{global_name}/w0_{i}_{raw_w}{n}.npy',
        w1=f'{output}/data/{global_name}/w1_{i}_{raw_w}{n}.npy',
        metaData=f'{output}/data/{global_name}/metaData_{n}.pkl',
        weights=w,
        features=features,
        #weightFeature=weightFeature,
        label=i,
        plot=True,
        nentries=n,
        #TreeName=treename,
        #pathA=p+nominal+".root",
        #pathB=p+variation+".root",
        global_name=global_name,
        plot_ROC=opts.plot_ROC,
        plot_obs_ROC=opts.plot_obs_ROC,
        ext_binning = binning,
        normalise = normalise,
        scaling=scale_method,
        plot_resampledRatio=opts.plot_resampledRatio,
    )
# Evaluate performance
print("<evaluate.py::__init__>::   Evaluate Performance of Model")
carl.evaluate_performance(
    x=f'{output}/data/{global_name}/X_val_{n}.npy',
    y=f'{output}/data/{global_name}/y_val_{n}.npy',
)
