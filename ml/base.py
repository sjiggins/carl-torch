from __future__ import absolute_import, division, print_function

import logging
import os
import json
import numpy as np
import pickle
import torch
import tarfile
import onnxruntime as ort
import onnx as onnx


from .utils.tools import create_missing_folders, load_and_check
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

logger = logging.getLogger(__name__)
class Estimator(object):
    """
    Abstract class for any ML estimator.
    Each instance of this class represents one neural estimator. The most important functions are:
    * `Estimator.train()` to train an estimator.
    * `Estimator.evaluate()` to evaluate the estimator.
    * `Estimator.save()` to save the trained model to files.
    * `Estimator.load()` to load the trained model from files.
    Please see the tutorial for a detailed walk-through.
    """

    def __init__(self, features=None, n_hidden=(100,), activation="tanh", dropout_prob=0.0):
        self.features = features
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout_prob = dropout_prob

        self.model = None
        self.n_observables = None
        self.n_parameters = None
        self.x_scaling_means = None
        self.x_scaling_stds = None
        self.scaling_method = None
        self.scaling_clamp = False
        self.clamp_min = None
        self.clamp_max = None
        self.x_scaling_quantile_up = None
        self.x_scaling_quantile_down = None

        self.divisions = 100 # binning for inputs if requested

    def train(self, *args, **kwargs):
        raise NotImplementedError


    def evaluate_ratio(self, *args, **kwargs):
        """
        Ratio estimation. Signature depends on the type of estimator. The first returned value is the ratio with
        shape `(n_thetas, n_x)` or `(n_x)`.
        """
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, filename, x, metaData, save_model=False, export_model=False, noTar=True):

        """
        Saves the trained model to four files: a JSON file with the settings, a pickled pyTorch state dict
        file, and numpy files for the mean and variance of the inputs (used for input scaling).
        Also exports model to onnx if export_model is set to True.
        Parameters
        ----------
        filename : str
            Path to the files. '_settings.json' and '_state_dict.pl' will be added.
        save_model : bool, optional
            If True, the whole model is saved in addition to the state dict. This is not necessary for loading it
            again with Estimator.load(), but can be useful for debugging, for instance to plot the computational graph.
        export_model : bool, optional
            If True, the whole model is exported to .onnx format to be loaded within a C++ envirnoment.
        Returns
        -------
            None
        """

        logger.info("Saving model to %s", filename)

        if self.model is None:
            raise ValueError("No model -- train or load model before saving!")

        # Check paths
        create_missing_folders([os.path.dirname(filename)])

        # Save settings
        logger.debug("Saving settings to %s_settings.json", filename)

        settings = self._wrap_settings()
        with open(filename + "_settings.json", "w") as f:
            json.dump(settings, f)
        # Save scaling
        if self.x_scaling_stds is not None and self.x_scaling_means is not None:
            logger.debug("Saving input scaling information to %s_x_means.npy and %s_x_stds.npy", filename, filename)
            np.save(filename + "_x_means.npy", self.x_scaling_means)
            np.save(filename + "_x_stds.npy", self.x_scaling_stds)
            np.save(filename + "_x_mins.npy", self.x_scaling_mins)
            np.save(filename + "_x_maxs.npy", self.x_scaling_maxs)

        # Save state dict
        logger.debug("Saving state dictionary to %s_state_dict.pt", filename)
        torch.save(self.model.state_dict(), filename + "_state_dict.pt")

        # Save model
        if save_model:
            logger.debug("Saving model to %s_model.pt", filename)
            torch.save(self.model, filename + "_model.pt")

        # Export model to onnx
        if export_model:
            self.model.eval()
            x = load_and_check(x)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dummy_input = torch.from_numpy(x[0].reshape(1, -1)).float().to(device)
            torch.onnx.export(self.model, dummy_input,filename+".onnx", export_params=True, input_names = ['input'],output_names = ['r_hat', 's_hat'], verbose = True)

        # Manipulate onnx model using 'onnxruntime' module directly
        #  Note: This is inefficient due to I/O reasons, however
        #        torch.onnx interface seemingly has no options for this
        if export_model and os.path.isfile(filename+".onnx"):

            ####################################
            ##        ONNXRUNTIME
            ## Example using Onnxruntime instead of Onxx
            ## Keeping only for prosperity for now
            ####################################
            ## Start the normal onnxruntime session using the model
            ## just saved
            #ort_session = ort.InferenceSession(filename+".onnx")
            ## Model Meta data
            #metaData = ort_session.get_modelmeta()
            ## Get the custom map
            #CustomMap = metaData.custom_metadata_map
            #print("Custom Meta-Data Map: {}".format(CustomMap))

            ## Define a new custom meta data map
            #CustomMap_new = {"Var1" : 200.0,
            #                 "Var2" : 5.0,
            #                 "Var3" : 1000.0,
            #                 "Var4" : 400.0,
            #                 "Var5" : 6.0,
            #             }
            #
            ## Load new custom map into mode
            #metaData.custom_metadata_map = CustomMap_new

            # Unable to save Onnx model from Onnxruntime Inference session it seems
            #   -> Makes sense given InferenceSession is designed to access and infer, not
            #      a data/model editor session.
            #ort_session.SaveModelMetadata() # Believe that this does not work
            ####################################
            ####################################

            ####################################
            ##        ONNX
            ####################################
            # Define a new custom meta data map
            #CustomMap_new = {"Var1" : 200.0,
            #                 "Var2" : 5.0,
            #                 "Var3" : 1000.0,
            #                 "Var4" : 400.0,
            #                 "Var5" : 6.0,
            #             }
            # Load model
            model = onnx.load(filename+".onnx")
            # Get Meta Data
            for index,(cust_key,cust_var) in enumerate(metaData.items()):
                meta = model.metadata_props.add()
                meta.key = cust_key
                meta.value = str(cust_var)
                # Check value
                logger.info(" New Meta data: %s ",model.metadata_props[index])


            # Save model
            onnx.save(model, filename+"_new"+".onnx")

            # Start the normal onnxruntime session using the model
            # just saved to check that the model was saved with the correct
            # metadata
            ort_session = ort.InferenceSession(filename+"_new"+".onnx")
            # Model Meta data
            metaData = ort_session.get_modelmeta()
            # Print Metadata
            CustomMap = metaData.custom_metadata_map
            logger.info(" Custom Meta-Data Map: %s",CustomMap)
            # Need to close the ort session for comleteness (C-style)
            ####################################
            ###################################


        # Tar model if training is done on GPU
        # tarfile in python is slow, so if noTar==True, skip this.
        if torch.cuda.is_available() and not noTar:
            tar = tarfile.open("models_out.tar.gz", "w:gz")
            for name in [filename+".onnx", filename + "_x_stds.npy", filename + "_x_means.npy",  filename + "_x_mins.npy",  filename + "_x_maxs.npy", filename + "_settings.json",  filename + "_state_dict.pt"]:
                tar.add(name)
            tar.close()

    def makeConfusion(self, filename, x,y):
        X = torch.from_numpy(x).type(torch.FloatTensor)
        y_pred = self.model(X)
        logger.info("acc %.2f",accuracy_score(y_pred,y))
        logger.info(confusion_matrix(y, y_pred))

    def load(self, filename, global_name=None, nentries=-1):

        """
        Loads a trained model from files.
        Parameters
        ----------
        filename : str
            Path to the files. '_settings.json' and '_state_dict.pl' will be added.
        Returns
        -------
            None
        """

        logger.info("Loading model from %s", filename)

        # Load settings and create model
        logger.debug("Loading settings from %s_settings.json", filename)
        with open(filename + "_settings.json", "r") as f:
            settings = json.load(f)
        self._unwrap_settings(settings)
        self._create_model()

        # Load scaling
        try:

            # Scale observables
            # Check if meta data has been saved, if so then scale using saved meta data from
            # initial total dataset loading stage
            metaData='data/'+global_name+'/metaData_'+str(nentries)+'.pkl'
            metaDataDict = None
            if os.path.exists(metaData):
                # Get the meta data containing the keys (input feature names)
                logger.info("Obtaining input features from metaData_{}.pkl".format(global_name))
                metaDataFile = open(metaData, 'rb')
                metaDataDict = pickle.load(metaDataFile)
                metaDataFile.close()

                # Initialise input scaling transformation
                self.initialize_input_transform(x=None, overwrite=False,
                                                metaData=metaDataDict, scaling=self.scaling_method)
            else:
                self.x_scaling_means = np.load(filename + "_x_means.npy")
                self.x_scaling_stds =  np.load(filename + "_x_stds.npy")
                self.x_scaling_mins =  np.load(filename + "_x_mins.npy")
                self.x_scaling_maxs =  np.load(filename + "_x_maxs.npy")
                logger.debug(
                    "  Found input scaling information: means %s, stds %s, mins %s, maxs %s  ", self.x_scaling_means, self.x_scaling_stds, self.x_scaling_mins, self.x_scaling_maxs
                )
        except FileNotFoundError:
            logger.warning("Scaling information not found in %s", filename)
            self.x_scaling_means = None
            self.x_scaling_stds = None
            self.x_scaling_mins = None
            self.x_scaling_maxs = None

        # Load state dict
        logger.debug("Loading state dictionary from %s_state_dict.pt", filename)
        self.model.load_state_dict(torch.load(filename + "_state_dict.pt", map_location="cpu")) # Likely an issue when in/on GPU mode/node

    def initialize_input_transform(self, x,
                                   transform=True, overwrite=True,
                                   metaData = None, scaling="minmax"):

        # Initially tell the user if values already set
        if (self.x_scaling_stds is not None  and
           self.x_scaling_means is not None and
           self.x_scaling_mins is not None  and
           self.x_scaling_maxs is not None
           and not overwrite):
            logger.info(
                "Input rescaling already defined. To overwrite, call initialize_input_transform(x, overwrite=True)."
            )
        elif transform and metaData is not None:
            # Now if the user passes metadata already use said metadata
            logger.info(
                "Input rescaling will run using metaData. To overwrite, call initialize_input_transform(x, overwrite=True, metaData=None).")

            np_pair_first = []
            np_pair_second = []
            for idx,(key,pair) in enumerate(metaData.items()):
                logger.info("   Passing for variable {}:".format(key))
                logger.info("        Pair = {}".format(pair))
                logger.info("        first = {}".format(pair[0]))
                logger.info("        second = {}".format(pair[-1]))
                np_pair_first.append(pair[0])
                np_pair_second.append(pair[-1])

            logger.info("Using scale method '{}'".format(scaling))
            # Assume the first entry is the first parameter (min/mean)
            # Assume the last entry is the second parameter (max/standard deviation)
            if scaling == "minmax":
                self.x_scaling_mins = np.array(np_pair_first)
                self.x_scaling_maxs = np.array(np_pair_second)
            elif scaling == "standard":
                self.x_scaling_means = np.array(np_pair_first)
                self.x_scaling_stds = np.array(np_pair_second)

            # Add the quantiles as it does not hurt at this stage
            if x is not None:
                self.x_scaling_quantile_down = np.quantile(x, 0, axis=0)
                self.x_scaling_quantile_up = np.quantile(x, 0.80, axis=0)
                if self.clamp_max is None:
                    self.clamp_max = self.x_scaling_quantile_up
                    if self.clamp_min is None:
                        self.clamp_min = self.x_scaling_quantile_down

        elif transform and metaData is None:
            logger.info("Setting up input rescaling")
            self.x_scaling_means = np.nanmean(x, axis=0)
            self.x_scaling_stds = np.maximum(np.nanstd(x, axis=0), 1.0e-6)
            self.x_scaling_mins = np.nanmin(x, axis=0)
            self.x_scaling_maxs = np.nanmax(x, axis=0)

            self.x_scaling_quantile_down = np.quantile(x, 0, axis=0)
            self.x_scaling_quantile_up = np.quantile(x, 0.80, axis=0)
            if self.clamp_max is None:
                self.clamp_max = self.x_scaling_quantile_up
            if self.clamp_min is None:
                self.clamp_min = self.x_scaling_quantile_down
        else:
            logger.info("Disabling input rescaling")
            n_parameters = x.shape[0]

            self.x_scaling_means = np.zeros(n_parameters)
            self.x_scaling_stds = np.ones(n_parameters)
            self.x_scaling_mins = np.zeros(n_parameters)
            self.x_scaling_maxs = np.ones(n_parameters)
            self.x_scaling_quantile_down = np.zeros(n_parameters)
            self.x_scaling_quantile_up = np.ones(n_parameters)

    def _clamp_inputs(self, x):
        logger.debug("<base.py::_transform_inputs()>::   Doing Clamping for inputs")
        # clamp value by 25% to 75% quntile
        if isinstance(x, torch.Tensor):
            clamp_max = torch.tesnor(self.clamp_max, dtype=x.dtype, device=x.device)
            clamp_min = torch.tensor(self.clamp_min, dtype=x.dtype, device=x.device)
            return torch.clamp(x, min=clamp_min, max=clamp_max)
        else:
            clamp_max = self.clamp_max
            clamp_min = self.clamp_min
            return np.clip(x, clamp_min, clamp_max)

    def _transform_inputs(self, x, scaling = "minmax"):
        # use the self.scaling method to overwritten the scaling arugmuent
        # i.e if self.scaling_method = None, scaling will be used.
        scaling  = self.scaling_method or scaling
        if scaling == "standard":
            #Check for standard deviation = 0 and none values
            if self.x_scaling_means is not None and self.x_scaling_stds is not None:
                logger.debug("Doing Standard Scaling")
                if isinstance(x, torch.Tensor):
                    x_scaled = x - torch.tensor(self.x_scaling_means, dtype=x.dtype, device=x.device)
                    x_scaled = x_scaled / torch.tensor(self.x_scaling_stds, dtype=x.dtype, device=x.device)
                else:
                    x_scaled = x - self.x_scaling_means
                    x_scaled /= self.x_scaling_stds

                # Check for nans/nums
                # -1 might not be best option if this is a valid value for a feature. Best option is to set to -10% of range below min or above max
                x_scaled = np.nan_to_num(x_scaled, nan=-1.0, posinf=0.0, neginf=0.0)
                #x_scaled = torch.tensor(np.nan_to_num(x_scaled, nan=-1.0, posinf=0.0, neginf=0.0), dtype=x.dtype, device=x.device )  # For GPU shenanigans
                #x_scaled = torch.tensor(x_scaled, dtype=x_scaled.dtype, device=x.device)
            else:
                logger.debug("Unable to do standard scaling")
                x_scaled = x
        else:
            # Check for none and 0 values
            if self.x_scaling_mins is not None and self.x_scaling_maxs is not None:
                logger.debug("Doing min-max scaling")
                logger.debug("self.x_scaling_mins = {}".format(self.x_scaling_mins))
                logger.debug("self.x_scaling_maxs = {}".format(self.x_scaling_maxs))
                if self.scaling_clamp:
                    x = self._clamp_inputs(x)
                if isinstance(x, torch.Tensor):
                    x_scaled = (x-torch.tensor(self.x_scaling_mins, dtype=x.dtype, device=x.device))
                    x_scaled = x_scaled/(torch.tensor(self.x_scaling_maxs, dtype=x.dtype, device=x.device) - torch.tensor(self.x_scaling_mins, dtype=x.dtype, device=x.device))
                else:
                    x_scaled = (x - self.x_scaling_mins)
                    #x_scaled = x_scaled/(self.x_scaling_maxs - self.x_scaling_mins)
                    diff = (self.x_scaling_maxs - self.x_scaling_mins)
                    x_scaled = np.divide(x_scaled, diff, out=np.zeros_like(x_scaled), where=diff!=0)
            else:
                logger.warning("Unable to do min-max scaling")
                x_scaled = x
            # Check for nans/nums and assigning dummy values
            # -1 might not be best option if this is a valid value for a feature. Best option is to set to -10% of range below min or above max
            x_scaled = np.nan_to_num(x_scaled, nan=-1.0, posinf=0.0, neginf=0.0)

        return x_scaled

    def _wrap_settings(self):
        settings = {
            "n_observables": self.n_observables,
            "n_parameters": self.n_parameters,
            "features": self.features,
            "n_hidden": list(self.n_hidden),
            "activation": self.activation,
            "dropout_prob": self.dropout_prob,
        }
        return settings

    def _unwrap_settings(self, settings):
        try:
            _ = str(settings["estimator_type"])
        except KeyError:
            raise RuntimeError(
                "Can't find estimator type information in file. Maybe this file was created with"
                " an incompatible MadMiner version < v0.3.0?"
            )

        self.n_observables = int(settings["n_observables"])
        self.n_hidden = tuple([int(item) for item in settings["n_hidden"]])
        self.activation = str(settings["activation"])
        self.features = settings["features"]
        if self.features == "None":
            self.features = None
        if self.features is not None:
            self.features = list([int(item) for item in self.features])

        try:
            self.dropout_prob = float(settings["dropout_prob"])
        except KeyError:
            self.dropout_prob = 0.0
            logger.info(
                "Can't find dropout probability in model file. Probably this file was created with an older"
            )

    def _create_model(self):
        raise NotImplementedError
