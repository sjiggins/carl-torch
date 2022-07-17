import sys
import importlib
from collections import OrderedDict

# ======================================================
# Dynamic import performed in the BuildAlgorithms() method
#from .utils.kde import KDE
#from .utils.pdes import PDES
# ======================================================

class SampleAugmentor():
    """
    Sample Augmentor base class that runs all data augmentations.
    This class is simply designed as a interface to all algorithms that
    augment the data input by the user during training. At present the 
    supported augmentation algorithms are:
    
    1)   Kernel Density Estimation = ml/utils/kde.py
    2)   Positive Density Estimation Sampler = ml/utils/pdes.py
    3)   Neural Re-sampling Estimator = ml/utils/nre.py
    """

    # Default initialisation algorithm
    def __init__(self, alg="",
                 x, features, metadata
                 weights=None, weight_name=None,
                 *args, **kwargs):    

        # Store the algorithms to be instantiated
        self.algs=alg.split(",")
    
        # Instantiate the relevant algorithm
        self.BuildAlg(self.algs)

        # Initialise the algorithms booked for running
        
        
    # Default fit class that executes the algorithm
    def Fit(self):
        pass

    # Default evaluation method for executing the algorithm and
    # applying the intended consequences of the algorithm to the 
    # data
    def Evaluate(self):
        pass

    # Plot method for creating diagnostic data about the algorithm 
    # execution
    def Plot(self):
        pass

    # Algorithm builder for SampleAugmentor class that imports the module
    # and then instantiates a class
    def BuildAugmentor(self):
        
        # Create an ordered dictionary that will be filled
        # in the order of the algorithm squence defied by
        # the user command line
        self.modules = OrderedDict()
        self.augmentors = OrderedDict()
        
        for algorithm in self.algs:
            # Modules are lower case
            self.modules[algorithm] = importlib.import_module(algorithm)
            # Classes are upper case of the module name
            self.augmentors[algorithm] = getattr(self.modules[algorithm], algorithm.upper())
