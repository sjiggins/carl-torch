from __future__ import absolute_import, division, print_function

import os
import pickle
from collections import OrderedDict
from .utils.plotting import draw_weighted_distributions

import logging
import numpy as np
import torch
from collections import OrderedDict

from .evaluate import evaluate_ratio_model, evaluate_performance_model
from .models import RatioModel
from .functions import get_optimizer, get_loss
from .utils.tools import load_and_check
from .trainers import RatioTrainer
from .base import Estimator

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(__name__)
class RatioEstimator(Estimator):
    """
    Parameters
    ----------
    features : list of int or None, optional
        Indices of observables (features) that are used as input to the neural networks. If None, all observables
        are used. Default value: None.
    n_hidden : tuple of int, optional
        Units in each hidden layer in the neural networks.
        Default value: (100,).
    activation : {'tanh', 'sigmoid', 'relu'}, optional
        Activation function. Default value: 'tanh'.
    """

    def _generate_required_data_list(self, type):
        """
        Generate dict for required data, the key is the lookup reference for checking,
        and the values are used for hitting if keys are not found.
        """

        valid_types = {
            "required",
            "optional_train_data",
            "optional_val_data",
            "per_epoch_plot",
            "per_epoch_save",
        }
        if type not in valid_types:
            raise TypeError(f"Unable to generate type checking for {type}")

        # minimum required data for the training:
        if type == "required":
            return {
                "X_train" : "prepared training data",
                "y_train" : "targe values",
                "w_train" : "event weights for for the training data",
                "features": "list of features used in the training",
            }
        elif type == "optional_train_data":
            return {f"{prefix}_train" : f"{prefix} training set" for prefix in ["X0", "X1", "w0", "w1"]}
        elif type == "optional_val_data":
            return {f"{prefix}_val" : f"{prefix} val set" for prefix in ["X0", "X1", "w0", "w1"]}
        # per epoch plotting required data
        elif type == "per_epoch_plot":
            return {
                "metaData" : "pickle file that contains meta data",
                "features" : "list of features used in the training",
                "plot" : "bool, enable plots saving",
                "nentries" : "int, number of input events",
                "global_name" : "str, name of the varition, e.g QSF4",
                "ext_binning" : "path to external YAML binning file",
                "verbose" : "bool, verbosity of plotting",
                "plot_ROC" : "bool, enable ROC curve",
                "plot_obs_ROC" : "bool, enbale ROC curve for observables",
                "normalise" : "bool, normalize distribution in plotting",
            }
        # per epoch model saving
        elif type == "per_epoch_save":
            return {
                "filename" : "prefix of the output model file",
                "metaData" : "pickle file that contains meta data",
                "save_model" : "bool, enable model saving to .pt format",
                "export_model" : "bool, enable model exporting to .oonx format",
            }
        else:
            raise TypeError(f"fail to generate check list for {type}")


    def check_required_data(self, input_data_dict, type):

        if not isinstance(input_data_dict, dict):
            raise TypeError(f"Argument input_data_dict needs to be type 'dict', but received {type(input_data_dict)}")

        generated_check = self._generate_required_data_list(type)

        if type == "required":
            if not all(x in input_data_dict for x in generated_check.keys()):
                raise KeyError(f"Unable to look up all required data, please have at least prepared {generated_check}")
        elif type == "optional_train_data" or type == "optional_val_data":
            if not all(x in input_data_dict for x in generated_check.keys()):
                logger.warning(f"unable to find all of the optional data {generated_check.keys()}")
                return False
        else:
            if type not in input_data_dict:
                if type == "per_epoch_plot":
                    logger.warning("Cannot enable per epoch plotting")
                elif type == "per_epoch_save":
                    logger.warning("Cannot enable per epoch saving of model")
                logger.warning(f"Pleast provide a lookup key '{type}' for 'input_data_dict' with the following dict format")
                logger.warning(f"{generated_check}")
                return False
            else:
                if not all(x in input_data_dict[type] for x in generated_check.keys()):
                    logger.warning(f"Cannot find all of the required keys for '{type}' from {input_data_dict[type]}")
                    logger.warning(f"Require {generated_check}")
                    return False

        logger.info(f"Pass data checking for {type=}")
        return True

    def train(
        self,
        method,
        input_data_dict,
        alpha=1.0,
        optimizer="amsgrad",
        optimizer_kwargs=None,
        n_epochs=50,
        batch_size=128,
        initial_lr=0.001,
        final_lr=0.0001,
        nesterov_momentum=None,
        validation_split=0.25,
        early_stopping=True,
        scale_inputs=True,
        limit_samplesize=None,
        memmap=False,
        verbose="some",
        scale_parameters=False,
        n_workers=8,
        clip_gradient=None,
        early_stopping_patience=None,
        intermediate_train_plot=None,
        intermediate_save=None,
        intermediate_stats_dist=False,
        stats_method_list = [],
        global_name="",
        plot_inputs=False,
        nentries=-1,
        loss_type="regular",
    ):

        """
        Trains the network.
        Parameters
        ----------
        method : str
            The inference method used for training. Allowed values are 'alice', 'alices', 'carl', 'cascal', 'rascal',
            and 'rolr'.
        x : ndarray or str
            Observations, or filename of a pickled numpy array.
        y : ndarray or str
            Class labels (0 = numeerator, 1 = denominator), or filename of a pickled numpy array.
        alpha : float, optional
            Default value: 1.
        optimizer : {"adam", "amsgrad", "sgd"}, optional
            Optimization algorithm. Default value: "amsgrad".
        n_epochs : int, optional
            Number of epochs. Default value: 50.
        batch_size : int, optional
            Batch size. Default value: 128.
        initial_lr : float, optional
            Learning rate during the first epoch, after which it exponentially decays to final_lr. Default value:
            0.001.
        final_lr : float, optional
            Learning rate during the last epoch. Default value: 0.0001.
        nesterov_momentum : float or None, optional
            If trainer is "sgd", sets the Nesterov momentum. Default value: None.
        validation_split : float or None, optional
            Fraction of samples used  for validation and early stopping (if early_stopping is True). If None, the entire
            sample is used for training and early stopping is deactivated. Default value: 0.25.
        early_stopping : bool, optional
            Activates early stopping based on the validation loss (only if validation_split is not None). Default value:
            True.
        scale_inputs : bool, optional
            Scale the observables to zero mean and unit variance. Default value: True.
        memmap : bool, optional.
            If True, training files larger than 1 GB will not be loaded into memory at once. Default value: False.
        verbose : {"all", "many", "some", "few", "none}, optional
            Determines verbosity of training. Default value: "some".
        Returns
        -------
            None
        """

        logger.info("Starting training")
        logger.info("  PyTorch version:                 %s", torch.__version__)
        logger.info("  Method:                 %s", method)
        logger.info("  Batch size:             %s", batch_size)
        logger.info("  Optimizer:              %s", optimizer)
        logger.info("  Optimizer kwargs:         {}".format(optimizer_kwargs))
        logger.info("  Epochs:                 %s", n_epochs)
        logger.info("  Learning rate:          %s initially, decaying to %s", initial_lr, final_lr)
        if optimizer == "sgd":
            logger.info("  Nesterov momentum:      %s", nesterov_momentum)
        logger.info("  Validation split:       %s", validation_split)
        logger.info("  Early stopping:         %s", early_stopping)
        logger.info("  Early stopping patience:         %s", early_stopping_patience)
        logger.info("  Scale inputs:           %s", scale_inputs)
        if limit_samplesize is None:
            logger.info("  Samples:                all")
        else:
            logger.info("  Samples:                %s", limit_samplesize)
        logger.info(f"  N hidden:                 {self.n_hidden}")
        logger.info(f"  Input loss type:                 {loss_type}")

        # checking data
        self.check_required_data(input_data_dict, "required")
        pass_opt_data_check = self.check_required_data(input_data_dict, "optional_train_data")
        pass_opt_data_check |= self.check_required_data(input_data_dict, "optional_val_data")
        if not pass_opt_data_check:
            intermediate_stats_dist = False
            intermediate_train_plot = False
            intermediate_save = False
        if intermediate_train_plot:
            intermediate_train_plot = self.check_required_data(input_data_dict, "per_epoch_plot")
        if intermediate_save:
            intermediate_save = self.check_required_data(input_data_dict, "per_epoch_save")

        # Load training data
        logger.info("Loading training data")
        load_and_check_list = ["X", "y", "w", "X0", "X1", "w0", "w1", "spec_x0", "spec_x0"]
        memmap_threshold = 1.0 if memmap else None
        for lookup_prefix in load_and_check_list:
            for lookup_suffix in ["train", "val"]:
                lookup = f"{lookup_prefix}_{lookup_suffix}"
                if lookup in input_data_dict:
                    checking = input_data_dict[lookup]
                    input_data_dict[lookup] = load_and_check(checking, memmap_files_larger_than_gb=memmap_threshold, name=lookup)

        # using old variables here to minimized changes below, might need to clean this up in the future
        x = input_data_dict.get("X_train")
        y = input_data_dict.get("y_train")
        w = input_data_dict.get("w_train")

        x0 = input_data_dict.get("X0_train", None)
        w0 = input_data_dict.get("w0_train", None)
        x1 = input_data_dict.get("X1_train", None)
        w1 = input_data_dict.get("w1_train", None)

        x_val = input_data_dict.get("X_val", None)
        y_val = input_data_dict.get("y_val", None)
        w_val = input_data_dict.get("w_val", None)

        # Infer dimensions of problem
        n_samples = x.shape[0]
        n_observables = x.shape[1]
        logger.info("Found %s samples with %s observables", n_samples, n_observables)

        # check validation dataset.
        # this is optional and require 'X_val', 'y_val', "w_val" in the input_data_dict
        external_validation = x_val is not None and y_val is not None
        if external_validation:
            logger.info("Found %s separate validation samples", x_val.shape[0])
            assert x_val.shape[1] == n_observables

        # trying to load metadata
        metaDataDict = input_data_dict.get("metaData", None)
        metaData=f"data/{global_name}/metaData_{nentries}.pkl"
        if metaDataDict is None and os.path.exists(metaData):
            with open(metaData, "rb") as metaDataFile:
                metaDataDict = pickle.load(metaDataFile)
                input_data_dict["metaData"] = metaDataDict

        # check initial plotting of input training
        plot_inputs = plot_inputs and all([_x is not None for _x in [x0, w0, x1, w1]])
        plot_inputs = plot_inputs and metaDataDict is not None

        # Scale features
        if scale_inputs:
            self.initialize_input_transform(x, overwrite=False)
            x = self._transform_inputs(x)
            if external_validation:
                x_val = self._transform_inputs(x_val)
            # If requested by user then transformed inputs are plotted
            if plot_inputs:
                logger.info(f"Plotting transformed input features for {global_name}")
                if metaDataDict:
                    # Get the meta data containing the keys (input feature anmes)
                    logger.info(f"Obtaining input features from metaData {metaData}")

                    # Transform the input data for x0, and x1
                    x0 = self._transform_inputs(x0)
                    x1 = self._transform_inputs(x1)

                    # Determine binning, and store in dicts
                    binning = OrderedDict()
                    minmax = OrderedDict()
                    for idx,(key,pair) in enumerate(metaDataDict.items()):
                        #  Integers values indicate well bounded data, so use full range
                        intTest = [ (i % 1) == 0  for i in x0[:,idx] ]
                        intTest = all(intTest) #np.all(intTest == True)
                        upperThreshold = 100 if intTest else 98
                        max = np.percentile(x0[:,idx], upperThreshold)
                        lowerThreshold = 0 if (np.any(x0[:,idx] < 0 ) or intTest) else 0
                        min = np.percentile(x0[:,idx], lowerThreshold)
                        minmax[idx] = [min,max]
                        binning[idx] = np.linspace(min, max, self.divisions)
                        logger.info(f"<loading.py::load_result>::Column {key}: {min=}, {max=}")
                    draw_weighted_distributions(
                        x0, x1,
                        w0, w1,
                        np.ones(w0.size),
                        metaDataDict.keys(),
                        binning,
                        "train-input", #label
                        global_name,
                        w0.size if w0.size < w1.size else w1.size,
                        True, #plot
                        None,
                    )

        else:
            self.initialize_input_transform(x, False, overwrite=False)

        # Features
        if self.features is not None:
            x = x[:, self.features]
            logger.info("Only using %s of %s observables", x.shape[1], n_observables)
            n_observables = x.shape[1]
            if external_validation:
                x_val = x_val[:, self.features]

        # Check consistency of input with model
        if self.n_observables is None:
            self.n_observables = n_observables

        if n_observables != self.n_observables:
            raise RuntimeError(
                "Number of observables does not match model: {} vs {}".format(n_observables, self.n_observables)
            )

        # Data
        data = self._package_training_data(method, x, y, w) #sjiggins - may be a problem if w = None
        if external_validation:
            data_val = self._package_training_data(method, x_val, y_val, w_val) #sjiggins
        else:
            data_val = None
        # Create model
        if self.model is None:
            logger.info("Creating model")
            self._create_model()
        # Losses
        # Note indeed the weight passed to get_loss will not be used
        # the loss_weights return from get_loss is from old implementation?
        # the packaged training set will have the weights carried along to RatioTrainer.forward_pass
        if w is None and x0 is not None and x1 is not None:
            w = len(x0)/len(x1)
            logger.info("Passing weight %s to the loss function to account for imbalanced dataset: ", w) #sjiggins
        loss_functions, loss_labels, loss_weights = get_loss(method, alpha, w, loss_type)

        # Optimizer
        opt, opt_kwargs = get_optimizer(optimizer, nesterov_momentum)
        # If optimizer_kwargs set by user then append
        #opt_kwargs = dict( opt_kwargs, optimizer_kwargs )
        if optimizer_kwargs is not None:
            opt_kwargs.update( optimizer_kwargs )

        # Train model
        logger.info("Training model")
        trainer = RatioTrainer(self.model, n_workers=n_workers)
        result = trainer.train(
            data=data,
            data_val=data_val,
            input_data_dict=input_data_dict,
            loss_functions=loss_functions,
            loss_weights=loss_weights, #sjiggins
            #loss_weights=w, #sjiggins
            loss_labels=loss_labels,
            epochs=n_epochs,
            batch_size=batch_size,
            optimizer=opt,
            optimizer_kwargs=opt_kwargs,
            initial_lr=initial_lr,
            final_lr=final_lr,
            validation_split=validation_split,
            early_stopping=early_stopping,
            verbose=verbose,
            clip_gradient=clip_gradient,
            early_stopping_patience=early_stopping_patience,
            intermediate_train_plot = intermediate_train_plot,
            intermediate_save = intermediate_save,
            intermediate_stats_dist = intermediate_stats_dist,
            stats_method_list = stats_method_list,
            estimator = self, # just pass the RatioEstimator object itself for intermediate evaluate and save
        )
        return result

    def evaluate_ratio(self, x):
        """
        Evaluates the ratio as a function of the observation x.
        Parameters
        ----------
        x : str or ndarray
            Observations or filename of a pickled numpy array.
        Returns
        -------
        ratio : ndarray
            The estimated ratio. It has shape `(n_samples,)`.
        """
        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logger.debug("Loading evaluation data")
        x = load_and_check(x)

        # Scale observables
        x = self._transform_inputs(x, scaling=self.scaling_method)

        # Restrict features
        if self.features is not None:
            x = x[:, self.features]
        logger.debug("Starting ratio evaluation")
        r_hat, s_hat = evaluate_ratio_model(
            model=self.model,
            xs=x,
        )
        logger.debug("Evaluation done")
        return r_hat, s_hat

    def evaluate(self, *args, **kwargs):
        return self.evaluate_ratio(*args, **kwargs)

    def evaluate_performance(self, x, y):
        """
        Evaluates the performance of the classifier.
        Parameters
        ----------
        x : str or ndarray
            Observations.
        y : str or ndarray
            Target.
        """
        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logger.debug("Loading evaluation data")
        x = load_and_check(x)
        y = load_and_check(y)

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict features
        if self.features is not None:
            x = x[:, self.features]
        evaluate_performance_model(
            model=self.model,
            xs=x,
            ys=y,
        )
        logger.debug("Evaluation done")

    def _create_model(self):
        self.model = RatioModel(
            n_observables=self.n_observables,
            n_hidden=self.n_hidden,
            activation=self.activation,
            dropout_prob=self.dropout_prob,
        )
    @staticmethod
    def _package_training_data(method, x, y, w): #sjiggins
        data = OrderedDict()
        data["x"] = x
        data["y"] = y
        data["w"] = w #sjiggins
        return data

    def _wrap_settings(self):
        settings = super(RatioEstimator, self)._wrap_settings()
        settings["estimator_type"] = "double_parameterized_ratio"
        return settings

    def _unwrap_settings(self, settings):
        super(RatioEstimator, self)._unwrap_settings(settings)

        estimator_type = str(settings["estimator_type"])
        if estimator_type != "double_parameterized_ratio":
            raise RuntimeError("Saved model is an incompatible estimator type {}.".format(estimator_type))
