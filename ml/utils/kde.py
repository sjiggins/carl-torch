from sklearn.neighbors import KernelDensity
from sklearn.grid import GridSearchCV
import numpy as np
import pandas as pd

class KDE():
    """
    Kernel Density Estimation class for determining in a 
    non-parameteric way the probability density function 
    of the input dataset defined by the user

    Data Members:
       x = np.array of data, with n-rows and n-features
       bandwidth = starting bandwidth which will be optimised by the grid search
       opt_bandwidth = optimal band width for KDE 
       kernel = one of the possible kernels for the density estimation technique [gaussian, tophat, ...]
    """

    def __init__(self, x, features, metadata, 
                 weights=None, weight_name=None,
                 bandwidth = None, kernel='gaussian'
                 ensemble=False, default_n_evs=100000):
        super(KDE, self).__init__()
        # Instantiate the data members
        self.x=x
        self.features=feats
        self.metadata=metadata
        self.kernel=kernel
        self.bandwidth=bandwidth
        self.upper=None
        self.lower=None
        self.ensemble=ensemble  # Run an ensemble approach for averaging
        self.default_n_evts=default_n_evts
        self.negative_weights=False
        self.grid_points=10

        # Determine if the weights are assigned, if not assign default values of 1.0
        if weights is None:
            # Assign a new column with weights of 1.0
            self.x = x.assign(w='1.0') # pandas dataframe (copy return)
            self.weight_name = 'w'
        else:
            self.x = self.x.join(self.weights[weight_name]) # pandas dataframe (copy return)
            self.weight_name = weight_name

        # Sample events for the KDE from the features provided
        self.GenSampledData()

        # Convert dataframe to numpy array for ease of use with sklearn
        self.CreateKDEDataFormat()

        # If the bandwidth is not set by the user (i.e. is None) then run
        # the estimation based on the data density:
        #    Scotts Rule:                      n**(-1./(d+4))
        #    Scotts Rule (weighted data):      n_{eff}**(-1./(d+4))
        #    Silvermans Rule:                  (n *(d + 2)/ 4.)**(-1./(d + 4))
        #    Silvermans Rule (weighted data):  (n_{eff}*(d + 2)/4.)**(-1./(d + 4))
        # Where:
        #    n = number of data points
        #    n_{eff} = number of effective data points = sum(weights)^{2}/sum(weights^{2})
        #    d = number of dimensions in the data (i.e. number of features)
        self.Execute()

        # Set the upper and lower limit as well for the KDE
        #   -> Incase future data spans beyond the KDE
        self.BoundaryLimits()

    def GenSampledData(self):
        
        # Determine the number of positive and negative events as
        # these need to be subsampled independently.
        self.negative_weights = (self.x[weight_name].values < 0).any()

        # Determine the number to sample
        if not self.negative_weight:
            self.n_evts['pos'] = len(self.x.index) if len(self.x.index) < self.default_n_evts else self.default_n_evts 
        else:
            n_temp_neg_evts = len(self.x.query("{} < 0".format(weight_name)).index) 
            n_temp_pos_evts = len(self.x.query("{} >= 0".format(weight_name)).index) 
            self.n_evts['neg'] = n_temp_neg_evts if n_temp_neg_evts < self.default_n_evts else self.default_n_evts
            self.n_evts['pos'] = n_temp_pos_evts if n_temp_pos_evts < self.default_n_evts else self.default_n_evts

        if self.negative_weights:
            #self.subsample['neg'] = (self.x.query("{} < 0".format(weight_name)))[ [feat for feat in features] ].sample(n=self.n_evts, weights=weight_name, random_state=1)
            self.subsample['neg'] = (self.x.query("{} < 0".format(weight_name))).sample(n=self.n_evts, weights=weight_name, random_state=1)
            
        # Subsample from the x features using the weight dataframe
        #self.subsample['pos'] = self.x[ [feat for feat in features] ].sample(n=self.n_evts, weights=weight_name, random_state=1)  
        self.subsample['pos'] = self.x.sample(n=self.n_evts, weights=weight_name, random_state=1)  

        # Calculate the bandwidth for each sample
        self.CalculateBandwidth()

        # Now remove the weight column from the dataset
        for key,sample in self.subsample.items():
            sample = sample[ [feat for feat in features] ]

        # We can not combine the positive and negative event weights because
        # the negative weights will be treated by a positive kernel
        #self.subsample[feat] = pd.concat(feat_neg, feat_pos).sample(frac=1).reset_index(drop=True)

            
        
        
    def CreateKDEDataFormat(self):
        # Convert the dataset from pandas to numpy
        for key,df in subsample.items():
            self.subsample_np[key] = df.to_numpy()


    def Execute(self):
        
        for key,sample in self.subsamples.items():

            # Create the N-dimensional mesh for the grid search
            bandwidths = np.meshgrid( *[np.linspace(bndwdt*0.5, bndwdt*2, self.grid_points) for feat_num in range(len(self.features)) ] ) # Using list comprehension here but not using the loop arg
            # Define the grid search
            self.grid[key] = GridSearchCV(KernelDensity(kernel=self.kernel),
                                {'bandwidth': bandwidths})
            
            
            # Fit the grid
            self.grid[key].fit(sample)

            # Get the best bandwidth
            self.opt_bandwidth[key] = self.grid[key].best_params_

            # Now assign the best KDE
            self.kde[key] = KernelDensity(kernel='gaussian', bandwidth=self.opt_bandwidth[key]).fit(sample)


    def BoundaryLimits(self):
        #temp_upper = np.max(x)
        for feat in self.features:
            self.upper[feat] = x[feat].max()
            self.lower[feat] = x[feat].min()


    def CalculateBandwidth(self):

        # Using general rule of thumbs for the moment
        #    Scotts Rule:                      n**(-1./(d+4))
        #    Scotts Rule (weighted data):      n_{eff}**(-1./(d+4))
        #    Silvermans Rule:                  (n *(d + 2)/ 4.)**(-1./(d + 4))
        #    Silvermans Rule (weighted data):  (n_{eff}*(d + 2)/4.)**(-1./(d + 4))
        # Where:
        #    n = number of data points
        #    n_{eff} = number of effective data points = sum(weights)^{2}/sum(weights^{2})
        #    d = number of dimensions in the data (i.e. number of features)

        # Scotts Rule - Weighted version (when w_{i}=1.0 it is the same as the unweighted version)
        for key,sample in self.subsample.items():
            self.bandwidth[key] = ( sample[weight_name].sum() ) ** ( -1.0/ (len(features)+4) )
