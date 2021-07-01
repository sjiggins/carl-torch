import os
import sys
import logging
import argparse
import torch
import tarfile
import pickle
import pathlib
import numpy as np
from ml import RatioEstimator
from ml import Loader
import numpy as np
from itertools import repeat


logger = logging.getLogger(__name__)


#################################################
# Arugment parsing
parser = argparse.ArgumentParser(usage="usage: %(prog)s [opts]")
parser.add_argument('--version', action='version', version='%(prog)s 1.0')
parser.add_argument('-n', '--nominal',   action='store', type=str, dest='nominal',   default='', help='Nominal sample name (root file name excluding the .root extension)')
parser.add_argument('-v', '--variation', action='store', type=str, dest='variation', default='', help='Variation sample name (root file name excluding the .root extension)')
parser.add_argument('-e', '--nentries',  action='store', type=int, dest='nentries',  default=1000, help='specify the number of events to do the training on, None means full sample')
parser.add_argument('-p', '--datapath',  action='store', type=str, dest='datapath',  default='./Inputs/', help='path to where the data is stored')
parser.add_argument('-g', '--global_name',  action='store', type=str, dest='global_name',  default='Test', help='Global name for identifying this run - used in folder naming and output naming')
parser.add_argument('-f', '--features',  action='store', type=str, dest='features',  default='', help='Comma separated list of features within tree')
parser.add_argument('-w', '--weightFeature',  action='store', type=str, dest='weightFeature',  default='DummyEvtWeight', help='Name of event weights feature in TTree')
parser.add_argument('-t', '--TreeName',  action='store', type=str, dest='treename',  default='Tree', help='Name of TTree name inside root files')
parser.add_argument('-b', '--binning',  action='store', type=str, dest='binning',  default=None, help='path to binning yaml file.')
parser.add_argument('-l', '--layers', action='store', type=int, dest='layers', nargs='*', default=None, help='number of nodes for each layer')
parser.add_argument('--batch',  action='store', type=int, dest='batch_size',  default=4096, help='batch size')
parser.add_argument('--per-epoch-plot', action='store_true', dest='per_epoch_plot', default=False, help='plotting train/validation result per epoch.')
parser.add_argument('--per-epoch-save', action='store_true', dest='per_epoch_save', default=False, help='saving trained model per epoch.')
parser.add_argument('--nepoch', action='store', dest='nepoch', type=int, default=500, help='Total number of epoch for training.')
parser.add_argument('--scale-method', action='store', dest='scale_method', type=str, default=None, help='scaling method for input data. e.g minmax, standard.')
opts = parser.parse_args()
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
#################################################

#################################################
# Loading of data from root of numpy arrays
loading = Loader()

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
#    tar = tarfile.open("data_out.tar.gz", "r:gz")
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
        normalise=False,
        debug=False,
    )
    logger.info(" Loaded new datasets ")
#######################################

#######################################
# Estimate the likelihood ratio using a NN model
#   -> Calculate number of input variables as rudimentary guess
structure = ( (len(features)*3, ) * 5)
# Use the number of inputs as input to the hidden layer structure
estimator = RatioEstimator(
    n_hidden=(structure),
    activation="relu"
)
estimator.scaling_method = scale_method

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
    }
    vali_args = {
        "x0":f'data/{global_name}/X0_val_{n}.npy',
        "x1":f'data/{global_name}/X1_val_{n}.npy',
        "w0":f'data/{global_name}/w0_val_{n}.npy',
        "w1":f'data/{global_name}/w1_val_{n}.npy',
        "metaData":metaData,
        "features":features,
        "label":"train",
        "plot":True,
        "nentries":n,
        "global_name":global_name,
        "ext_binning":binning,
        "verbose" : False,
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

# perform training
train_loss, val_loss, accuracy_train, accuracy_val = estimator.train(
    method='carl',
    batch_size=batch_size,
    n_epochs=nepoch,
    validation_split=0.25,
    x=x,
    y=y,
    w=w,
    x0=x0,
    x1=x1,
    scale_inputs=True,
    early_stopping=False,
    intermediate_train_plot = intermediate_train_plot,
    intermediate_save = intermediate_save,
)

# saving loss values and final trained models
np.save(f"loss_train_{global_name}.npy", train_loss)
np.save(f"loss_val_{global_name}.npy", val_loss)
np.save(f"accuracy_train_{global_name}.npy", accuracy_train)
np.save(f"accuracy_val_{global_name}.npy", accuracy_val)
estimator.save('models/'+ global_name +'_carl_'+str(n), x, metaData, export_model = True, noTar=True)
########################################
