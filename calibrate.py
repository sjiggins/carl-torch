import optparse
import os
import sys
import logging
from ml import RatioEstimator
from ml.utils.loading import Loader
from ml.calibration import CalibratedClassifier
from ml.base import Estimator
import numpy as np


#parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
#parser.add_option('-s', '--samples',   action='store', type=str, dest='samples',   default='dilepton', help='samples to derive weights for. Sherpa 2.2.8 ttbar dilepton')
#parser.add_option('-v', '--variation', action='store', type=str, dest='variation', default='QSFUP', help='variation to derive weights for. default QSF down to QSF up')
#parser.add_option('-n', '--nentries',  action='store', type=str, dest='nentries',  default=1000, help='specify the number of events do do the training on, default None means full sample')
#parser.add_option('-p', '--datapath',  action='store', type=str, dest='datapath',  default='/eos/atlas/unpledged/group-tokyo/users/tatsuya/TruthAOD/Temp/Tuples/', help='path to where the data is stored')
#(opts, args) = parser.parse_args()
#sample  = opts.samples
#var = opts.variation
#n = opts.nentries
#p = opts.datapath
#loading = Loader()

###########################################
parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-n', '--nominal',   action='store', type=str, dest='nominal',   default='', help='Nominal sample name (root file name excluding the .root extension)')
parser.add_option('-v', '--variation', action='store', type=str, dest='variation', default='', help='Variation sample name (root file name excluding the .root extension)')
parser.add_option('-e', '--nentries',  action='store', type=str, dest='nentries',  default=1000, help='specify the number of events to do the training on, None means full sample')
parser.add_option('-p', '--datapath',  action='store', type=str, dest='datapath',  default='./Inputs/', help='path to where the data is stored')
parser.add_option('-g', '--global_name',  action='store', type=str, dest='global_name',  default='Test', help='Global name for identifying this run - used in folder naming and output naming')
parser.add_option('-f', '--features',  action='store', type=str, dest='features',  default='', help='Comma separated list of features within tree')
parser.add_option('-w', '--weightFeature',  action='store', type=str, dest='weightFeature',  default='', help='Name of event weights feature in TTree')
parser.add_option('-t', '--TreeName',  action='store', type=str, dest='treename',  default='Tree', help='Name of TTree name inside root files')
parser.add_option('--PlotROC',  action="store_true", dest='plot_ROC',  help='Flag to determine if one should plot ROC')
parser.add_option('--PlotObsROC',  action="store_true", dest='plot_obs_ROC',  help='Flag to determine if one should plot observable ROCs')
parser.add_option('--scale-method', action='store', dest='scale_method', type=str, default=None, help='scaling method for input data. e.g minmax, standard.')
(opts, args) = parser.parse_args()
nominal  = opts.nominal
variation = opts.variation
n = opts.nentries
p = opts.datapath
global_name = opts.global_name
features = opts.features.split(",")
weightFeature = opts.weightFeature
treename = opts.treename
scale_method=opts.scale_method
loading = Loader()
###########################################

logger = logging.getLogger(__name__)
if os.path.exists('data/'+global_name+'/X_train_'+str(n)+'.npy') and os.path.exists('data/'+global_name+'/metaData_'+str(n)+'.pkl'):
    logger.info(" Doing calibration of model trained with datasets: [{},{}], with {} events.", nominal, variation, n)
else:
    logger.info(" No datasets available for evaluation of model trained with datasets: [{},{}] with {} events.".format(nominal, variation, n))
    logger.info("ABORTING")
    sys.exit()

carl = RatioEstimator()
carl.scaling_method = scale_method
carl.load('models/'+global_name+'_carl_'+str(n),global_name=global_name,nentries=n)
#load
#evaluate = ['train']
X = 'data/'+global_name+'/X_train_'+str(n)+'.npy'
y = 'data/'+global_name+'/y_train_'+str(n)+'.npy'
w = 'data/'+global_name+'/w_train_'+str(n)+'.npy'
r_hat, s_hat = carl.evaluate(X)
r_hat = np.nan_to_num(r_hat, nan=1.0, posinf=1.0, neginf=1.0)
s_hat = np.nan_to_num(s_hat, nan=1.0, posinf=1.0, neginf=1.0)
# Temporary weight clipping
r_hat[r_hat > 1.0] = 1.0
r_hat[r_hat < 0.0] = 0.0
s_hat[s_hat > 1.0] = 1.0
s_hat[s_hat < 0.0] = 0.0


calib = CalibratedClassifier(carl, global_name=global_name)
calib.fit(X=X,y=y,w=w)
p0, p1, r_cal = calib.predict(X=X)
#w_cal = 1/r_cal
loading.load_calibration(y_true = y,
                         p1_raw = s_hat, 
                         p1_cal = p1, 
                         label = 'calibrated',
                         global_name = global_name,
                         plot = True,
)

evaluate = ['train', 'val']
#evaluate = ['val']
for i in evaluate:
    p0, p1, r_cal = calib.predict(X ='data/'+global_name+'/X0_'+i+'_'+str(n)+'.npy')
    w = 1./r_cal
    loading.load_result(x0='data/'+global_name+'/X0_'+i+'_'+str(n)+'.npy',     
                        x1='data/'+global_name+'/X1_'+i+'_'+str(n)+'.npy',
                        w0='data/'+global_name+'/w0_'+i+'_'+str(n)+'.npy',     
                        w1='data/'+global_name+'/w1_'+i+'_'+str(n)+'.npy',
                        metaData='data/'+global_name+'/metaData_'+str(n)+'.pkl',
                        weights=w, 
                        features=features,
                        #weightFeature=weightFeature,
                        label=i+"_calib",
                        plot=True,
                        nentries=n,
                        #TreeName=treename,
                        #pathA=p+nominal+".root",
                        #pathB=p+variation+".root",
                        global_name=global_name,
                        plot_ROC=opts.plot_ROC,
                        plot_obs_ROC=opts.plot_obs_ROC,
                    )
    #loading.load_result(x0='data/'+sample+'/'+var+'/X0_'+i+'_'+str(n)+'.npy',
    #                    x1='data/'+sample+'/'+var+'/X1_'+i+'_'+str(n)+'.npy',
    #                    weights=w, 
    #                    label=i+'_calib',
    #                    do=sample,
    #                    var=var,    
    #                    plot=True,
    #                    n=n,
    #                    path=p,
    #)

