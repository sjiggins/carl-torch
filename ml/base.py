from __future__ import absolute_import, division, print_function

import logging
import os
import json
import numpy as np
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

    def load(self, filename):

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
        self.model.load_state_dict(torch.load(filename + "_state_dict.pt", map_location="cpu"))

    def initialize_input_transform(self, x, transform=True, overwrite=True):
        if self.x_scaling_stds is not None and self.x_scaling_means is not None and self.x_scaling_mins is not None and self.x_scaling_maxs is not None and not overwrite:
            logger.info(
                "Input rescaling already defined. To overwrite, call initialize_input_transform(x, overwrite=True)."
            )
        elif transform:
            logger.info("Setting up input rescaling")
            self.x_scaling_means = np.mean(x, axis=0)
            self.x_scaling_stds = np.maximum(np.std(x, axis=0), 1.0e-6)
            self.x_scaling_mins = np.min(x, axis=0)
            self.x_scaling_maxs = np.max(x, axis=0)
        else:
            logger.info("Disabling input rescaling")
            n_parameters = x.shape[0]

            self.x_scaling_means = np.zeros(n_parameters)
            self.x_scaling_stds = np.ones(n_parameters)
            self.x_scaling_mins = np.zeros(n_parameters)
            self.x_scaling_maxs = np.ones(n_parameters)

    def _transform_inputs(self, x, scaling = "min-max"):

        #print("<base.py::_transform_inputs()>:: Converting dummy values into NaN")
        #print("feature type {}".format(x.dtype))
        #x_aux = x.astype(float) # Need to be float to convert values into NaN
        #x_aux[x_aux==-99999] = np.nan
        # use the self.scaling method to overwritten the scaling arugmuent
        # i.e if self.scaling_method = None, scaling will be used.
        scaling  = self.scaling_method or scaling
        if scaling == "standard":
            #Check for standard deviation = 0 and none values
            if self.x_scaling_means is not None and self.x_scaling_stds is not None:
                print("<base.py::_transform_inputs()>::   Doing Standard Scaling")
                if isinstance(x, torch.Tensor):
                    x_scaled = x - torch.tensor(self.x_scaling_means, dtype=x.dtype, device=x.device)
                    x_scaled = x_scaled / torch.tensor(self.x_scaling_stds, dtype=x.dtype, device=x.device)
                else:
                    x_scaled = x - self.x_scaling_means
                    x_scaled /= self.x_scaling_stds
            else:
                print("<base.py::_transform_inputs()>::   unable to do standard scaling")
                x_scaled = x
        else:
            # Check for none and 0 values
            if self.x_scaling_mins is not None and self.x_scaling_maxs is not None:
                print("<base.py::_transform_inputs()>::   Doing min-max scaling")
                if isinstance(x, torch.Tensor):
                    x_scaled = (x-torch.tensor(self.x_scaling_mins, dtype=x.dtype, device=x.device))
                    x_scaled = x_scaled/(torch.tensor(self.x_scaling_maxs, dtype=x.dtype, device=x.device) - torch.tensor(self.x_scaling_mins, dtype=x.dtype, device=x.device))
                else:
                    x_scaled = (x - self.x_scaling_mins)
                    #x_scaled = x_scaled/(self.x_scaling_maxs - self.x_scaling_mins)
                    diff = (self.x_scaling_maxs - self.x_scaling_mins)
                    x_scaled = np.divide(x_scaled, diff, out=np.zeros_like(x_scaled), where=diff!=0)
            else:
                print("<base.py::_transform_inputs()>::   unable to do min-max scaling")
                x_scaled = x

        print("<base.py::_transform_inputs()>:: Converting NaN to -1 values")
        x_scaled = np.nan_to_num(x_scaled, nan=-1) #After ignoring dummy values (NaN), these are converted into -1 values
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
