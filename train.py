import os
import sys
import logging
#import argparse
import tarfile
import pickle
import pathlib
import numpy as np
from arg_handler import arg_handler_train
from ml import RatioEstimator
from ml import Loader
from ml import Filter
import numpy as np
from itertools import repeat

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
weightFeature = opts.weightFeature
treename = opts.treename
binning = opts.binning
n_hidden = tuple(opts.layers) if opts.layers != None else tuple( repeat( (len(features)), 3) )
batch_size = opts.batch_size
per_epoch_plot = opts.per_epoch_plot
per_epoch_save = opts.per_epoch_save
nepoch = opts.nepoch
scale_method = opts.scale_method
weight_clipping = opts.weight_clipping
weight_sigma = opts.weight_nsigma
polarity = opts.polarity
loss_type = opts.loss_type
BoolFilter = opts.BoolFilter
#################################################

#################################################
# Loading of data from root of numpy arrays
loading = Loader()
if BoolFilter != None:
    InputFilter = Filter(FilterString = BoolFilter)
    loading.Filter= InputFilter

# Exception handling for input files - .root
if os.path.exists(p+nominal+'.root') or os.path.exists('data/'+global_name+'/X_train_'+str(n)+'.npy'):
    logger.info(" Doing training of model with datasets: %s with %s  events.", nominal, n)
else:
    logger.info(" Trying to do training of model with datasets: %s with %s  events.", nominal, n)
    logger.info(" This file or directory does not exist.")
    sys.exit()

if os.path.exists(p+variation+'.root') or os.path.exists('data/'+global_name+'/X_train_'+str(n)+'.npy'):
    logger.info(" Doing training of model with datasets: %s with %s  events.", variation, n)
else:
    logger.info(" Trying to do training of model with datasets: %s with %s  events.", variation, n)
    logger.info(" This file or directory does not exist.")
    sys.exit()

if os.path.exists(f"data/{global_name}/data_out.tar.gz"):
    # tar = tarfile.open("data_out.tar.gz", "r:gz")
    tar = tarfile.open(f"data/{global_name}/data_out.tar.gz")
    tar.extractall()
    tar.close()

# Check if already pre-processed numpy arrays exist
if os.path.exists('data/'+global_name+'/X_train_'+str(n)+'.npy'):
    logger.info(" Loaded existing datasets ")
    x='data/'+global_name+'/X_train_'+str(n)+'.npy'
    y='data/'+global_name+'/y_train_'+str(n)+'.npy'
    w='data/'+global_name+'/w_train_'+str(n)+'.npy'
    x0='data/'+global_name+'/X0_train_'+str(n)+'.npy'
    w0='data/'+global_name+'/w0_train_'+str(n)+'.npy'
    x1='data/'+global_name+'/X1_train_'+str(n)+'.npy'
    w1='data/'+global_name+'/w1_train_'+str(n)+'.npy'
    f = open('data/'+global_name+'/metaData_'+str(n)+".pkl","rb")
    metaData = pickle.load(f)
    f.close()
else:
    x, y, x0, x1, w, w0, w1, metaData = loading.loading(
        folder=f"{pathlib.Path('./data/').resolve()}/",
        plot=True,
        global_name=global_name,
        features=features,
        weightFeature=weightFeature,
        TreeName=treename,
        randomize=False,
        save=True,
        correlation=True,
        preprocessing=False,
        nentries=n,
        pathA=p+nominal+".root",
        pathB=p+variation+".root",
        noTar=True,
        normalise=opts.normalise_inputs,
        debug=False,
        weight_preprocess=weight_sigma > 0,
        weight_preprocess_nsigma=weight_sigma,
        large_weight_clipping=weight_clipping,
        weight_polarity=polarity,
        scaling=scale_method,
        algorithms=opts.algorithms.split(',') if opts.algorithms is not None else None,
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
intermediate_train_plot = None
intermediate_save = None
if per_epoch_plot:
    # arguments for training and validation sets for loading.load_result
    train_args = {
        "x0":x0,
        "x1":x1,
        "w0":w0,
        "w1":w1,
        "metaData":metaData,
        "features":features,
        "label":"train",
        "plot":True,
        "nentries":n,
        "global_name":global_name,
        "ext_binning":binning,
        "verbose" : False,
        "plot_ROC" : False,
        "plot_obs_ROC" : False,
        "normalise" : True, # plotting
    }
    vali_args = {
        "x0":f'data/{global_name}/X0_val_{n}.npy',
        "x1":f'data/{global_name}/X1_val_{n}.npy',
        "w0":f'data/{global_name}/w0_val_{n}.npy',
        "w1":f'data/{global_name}/w1_val_{n}.npy',
        "metaData":metaData,
        "features":features,
        "label":"val",
        "plot":True,
        "nentries":n,
        "global_name":global_name,
        "ext_binning":binning,
        "verbose" : False,
        "plot_ROC" : False,
        "plot_obs_ROC" : False,
        "normalise" : True,  # plotting
    }
    intermediate_train_plot = (
        (estimator.evaluate, {"train":x0, "val":f'data/{global_name}/X0_val_{n}.npy'}),
        (loading.load_result, {"train":train_args, "val":vali_args}),
    )
if per_epoch_save:
    intermediate_save_args = {
        "filename" : f"{global_name}_carl_{n}",
        "x" : x,
        "metaData" : metaData,
        "save_model" : True,
        "export_model" : True,
    }
    intermediate_save = (
        estimator.save, intermediate_save_args
    )


# additional options to pytorch training package
kwargs = {}
if opts.regularise is not None:
    logger.info("L2 loss regularisation included.")
    kwargs={"weight_decay": 1e-5}


# perform training
train_loss, val_loss, accuracy_train, accuracy_val = estimator.train(
    method='carl',
    batch_size=batch_size,
    n_epochs=nepoch,
    validation_split=0.25,
    #optimizer="amsgrad",
    x=x,
    y=y,
    w=w,
    x0=x0,
    x1=x1,
    w0=w0,
    w1=w1,
    scale_inputs=True,
    scaling=scale_method,
    early_stopping=False,
    #early_stopping_patience=20,
    intermediate_train_plot = intermediate_train_plot,
    intermediate_save = intermediate_save,
    optimizer_kwargs=kwargs,
    global_name=global_name,
    plot_inputs=False,    
    nentries=n,
    loss_type=loss_type,
    #initial_lr=0.0001,
    #final_lr=0.00001,
)

# saving loss values and final trained models
np.save(f"loss_train_{global_name}.npy", train_loss)
np.save(f"loss_val_{global_name}.npy", val_loss)
np.save(f"accuracy_train_{global_name}.npy", accuracy_train)
np.save(f"accuracy_val_{global_name}.npy", accuracy_val)
estimator.save('models/'+ global_name +'_carl_'+str(n), x, metaData, export_model = True, noTar=True)
########################################
