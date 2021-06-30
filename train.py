import os
import sys
import logging
import optparse
import torch
import tarfile
import pickle
from ml import RatioEstimator
from ml import Loader


#################################################
# Arugment parsing
parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-n', '--nominal',   action='store', type=str, dest='nominal',   default='', help='Nominal sample name (root file name excluding the .root extension)')
parser.add_option('-v', '--variation', action='store', type=str, dest='variation', default='', help='Variation sample name (root file name excluding the .root extension)')
parser.add_option('-e', '--nentries',  action='store', type=str, dest='nentries',  default=1000, help='specify the number of events to do the training on, None means full sample')
parser.add_option('-p', '--datapath',  action='store', type=str, dest='datapath',  default='./Inputs/', help='path to where the data is stored')
parser.add_option('-g', '--global_name',  action='store', type=str, dest='global_name',  default='Test', help='Global name for identifying this run - used in folder naming and output naming')
parser.add_option('-f', '--features',  action='store', type=str, dest='features',  default='', help='Comma separated list of features within tree')
parser.add_option('-w', '--weightFeature',  action='store', type=str, dest='weightFeature',  default='DummyEvtWeight', help='Name of event weights feature in TTree')
parser.add_option('-t', '--TreeName',  action='store', type=str, dest='treename',  default='Tree', help='Name of TTree name inside root files')
(opts, args) = parser.parse_args()
nominal  = opts.nominal
variation = opts.variation
n = opts.nentries
p = opts.datapath
global_name = opts.global_name
features = opts.features.split(",")
weightFeature = opts.weightFeature
treename = opts.treename
#################################################

#################################################
# Loading of data from root of numpy arrays
loading = Loader()
logger = logging.getLogger(__name__)

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

if os.path.exists("data_out.tar.gz"):
#    tar = tarfile.open("data_out.tar.gz", "r:gz")
    tar = tarfile.open("data_out.tar.gz")
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
        folder='./data/',
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
        normalise=True,
        debug=False,
    )
    logger.info(" Loaded new datasets ")
#######################################

#######################################
# Estimate the likelihood ratio
estimator = RatioEstimator(
    n_hidden=(11,11,11,11),
    activation="relu"
)
estimator.train(
    method='carl',
    batch_size=1024,
    n_epochs=500,
    early_stopping=True,
    validation_split=0.25,
    x=x,
    y=y,
    w=w,
    x0=x0,
    x1=x1,
    scale_inputs=True,
)
estimator.save('models/'+ global_name +'_carl_'+str(n), x, metaData, export_model = True)
########################################
