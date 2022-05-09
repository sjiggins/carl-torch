import numpy as np
import argparse
import pickle
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
from ml.utils.plotting import draw_weighted_distributions

# Define logger
logger = logging.getLogger(__name__)

if __name__ == "__main__":
   parser = argparse.ArgumentParser(usage="usage: %prog [opts]")
   parser.add_argument('--version', action='version', version='%prog 1.0')
   parser.add_argument('-g', '--global_name',  action='store', type=str, dest='global_name',  default='Test', help='Global name for identifying this run - used in folder naming and output naming')
   parser.add_argument('-e', '--nentries',  action='store', type=int, dest='nentries',  default=1000, help='specify the number of events to do the training on, None means full sample')
   parser.add_argument('-d', '--divisions', action='store', type=int, dest='divisions', default=100, help='specify the number of divisions in the plots')
   opts = parser.parse_args()
   
   # Get data from numpy datasets
   x0_train = f"data/{opts.global_name}/X0_train_{opts.nentries}.npy"
   x0_val = f"data/{opts.global_name}/X0_val_{opts.nentries}.npy"
   x1_train = f"data/{opts.global_name}/X1_train_{opts.nentries}.npy"
   x1_val = f"data/{opts.global_name}/X1_val_{opts.nentries}.npy"
   
   # Get the weights
   w0_train = f"data/{opts.global_name}/w0_train_{opts.nentries}.npy"
   w0_val = f"data/{opts.global_name}/w0_val_{opts.nentries}.npy"
   w1_train = f"data/{opts.global_name}/w1_train_{opts.nentries}.npy"
   w1_val = f"data/{opts.global_name}/w1_val_{opts.nentries}.npy"
   
   # Get the metadata
   metaData='data/'+opts.global_name+'/metaData_'+str(opts.nentries)+'.pkl'
   metaDataFile = open(metaData, 'rb')
   metaDataDict = pickle.load(metaDataFile)
   
   # Load the datasets
   x0_train_array = np.load(x0_train)
   x0_val_array = np.load(x0_val)
   x1_train_array = np.load(x1_train)
   x1_val_array = np.load(x1_val)
   w0_train_array = np.load(w0_train)
   w0_val_array = np.load(w0_val)
   w1_train_array = np.load(w1_train)
   w1_val_array = np.load(w1_val)
   
   # Normalise based on the metadata
   binning = defaultdict()
   minmax = defaultdict()
   for idx,(key,pair) in enumerate(metaDataDict.items()):
      
      #  Integers values indicate well bounded data, so use full range
      intTest = [ (i % 1) == 0  for i in x0_train_array[:,idx] ]
      intTest = all(intTest) #np.all(intTest == True)
      upperThreshold = 100 if intTest else 99
      max = np.nanpercentile(x0_train_array[:,idx], upperThreshold)
      lowerThreshold = 0 if (np.any(x0_train_array[:,idx] < 0 ) or intTest) else 0
      min = np.nanpercentile(x0_train_array[:,idx], lowerThreshold)
      minmax[idx] = [min,max]
      binning[idx] = np.linspace(min, max, opts.divisions)
      logger.info("Column {}:  min  =  {},  max  =  {}"
                  .format(key,min,max))
      
      # Remove nans
      x0_train_array[:,idx] = np.nan_to_num(x0_train_array[:,idx])
      x1_train_array[:,idx] = np.nan_to_num(x1_train_array[:,idx])
      x0_val_array[:,idx] = np.nan_to_num(x0_val_array[:,idx])
      x1_val_array[:,idx] = np.nan_to_num(x1_val_array[:,idx])
      
      
   # Now draw the metadata normalised data
   draw_weighted_distributions(x0_train_array, x0_val_array, 
                               w0_train_array, w0_val_array,
                               np.ones(w0_train_array.size),
                               metaDataDict.keys(),
                               binning,
                               "Val-Train-comp_x0", #label
                               opts.global_name, 
                               w0_train_array.size if w0_train_array.size < w0_val_array.size else w0_val_array.size, 
                               True, #plot
                               None)
   
   # Now draw the metadata normalised data
   draw_weighted_distributions(x1_train_array, x1_val_array, 
                               w1_train_array, w1_val_array,
                               np.ones(w1_train_array.size),
                               metaDataDict.keys(),
                               binning,
                               "Val-Train-comp_x1", #label
                               opts.global_name, 
                               w1_train_array.size if w1_train_array.size < w1_val_array.size else w1_val_array.size, 
                               True, #plot
                               None)
   
   
   #plt.plot(train_loss, label="train loss")
   #plt.plot(val_loss, label="val loss")
   #plt.ylabel("loss")
   #plt.legend(frameon=False, title="")
   ##plt.show()
   #plt.savefig("train_val_loss.png")
   
