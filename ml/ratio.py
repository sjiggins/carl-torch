from __future__ import absolute_import, division, print_function

import os
import pickle
from collections import defaultdict
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

    def train(
        self,
        method,
        x,
        y,
        w=None,
        x0=None,
        x1=None,
        w0=None,
        w1=None,
        x_val=None,
        y_val=None,
        w_val=None,
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

        # Load training data
        logger.info("Loading training data")
        memmap_threshold = 1.0 if memmap else None
        x  = load_and_check(x, memmap_files_larger_than_gb=memmap_threshold, name="features")
        y  = load_and_check(y, memmap_files_larger_than_gb=memmap_threshold, name="target")
        x0 = load_and_check(x0, memmap_files_larger_than_gb=memmap_threshold, name="nominal features")
        x1 = load_and_check(x1, memmap_files_larger_than_gb=memmap_threshold, name="variation features")
        w = load_and_check(w, memmap_files_larger_than_gb=memmap_threshold, name="weights")
        w0 = load_and_check(w0, memmap_files_larger_than_gb=memmap_threshold, name="weight1")
        w1 = load_and_check(w1, memmap_files_larger_than_gb=memmap_threshold, name="weights2")

        # Infer dimensions of problem
        n_samples = x.shape[0]
        n_observables = x.shape[1]
        logger.info("Found %s samples with %s observables", n_samples, n_observables)

        external_validation = x_val is not None and y_val is not None
        if external_validation:
            x_val = load_and_check(x_val, memmap_files_larger_than_gb=memmap_threshold, name="x_val")
            y_val = load_and_check(y_val, memmap_files_larger_than_gb=memmap_threshold, name="y_val")
            w_val = load_and_check(w_val, memmap_files_larger_than_gb=memmap_threshold, name="w_val")
            logger.info("Found %s separate validation samples", x_val.shape[0])

            assert x_val.shape[1] == n_observables

        # trying to load metadata
        metaDataDict = None
        metaData=f"data/{global_name}/metaData_{nentries}.pkl"
        if os.path.exists(metaData):
            with open(metaData, "rb") as metaDataFile:
                metaDataDict = pickle.load(metaDataFile)

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
                    binning = defaultdict()
                    minmax = defaultdict()
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
                        logger.info("<loading.py::load_result>::   Column {}:  min  =  {},  max  =  {}"
                              .format(key,min,max))
                    draw_weighted_distributions(x0, x1,
                                                w0, w1,
                                                np.ones(w0.size),
                                                metaDataDict.keys(),
                                                binning,
                                                "train-input", #label
                                                global_name,
                                                w0.size if w0.size < w1.size else w1.size,
                                                True, #plot
                                                None)

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
        if w is None:
            w = len(x0)/len(x1)
            logger.info("Passing weight %s to the loss function to account for imbalanced dataset: ", w) #sjiggins
        loss_functions, loss_labels, loss_weights = get_loss(method, alpha, w, loss_type)

        # Optimizer
        opt, opt_kwargs = get_optimizer(optimizer, nesterov_momentum)
        # If optimizer_kwargs set by user then append
        #opt_kwargs = dict( opt_kwargs, optimizer_kwargs )
        if optimizer_kwargs is not None:
            opt_kwargs.update( optimizer_kwargs )

        # prepare data for x0 and x1 separately for intermediate calculation
        # Note: for `data` argument in RatioTrainer.train,
        # maybe it's better to pass all the x0,x1,w0,x,y etc,
        # and run packaging within this method? This allows us to reuse part
        # of the data for intermediate calculation.
        if intermediate_stats_dist:
            feature_data = {
                "feature_names" : list(metaDataDict.keys()),
                "x0" : x0,
                "w0" : w0,
                "x1" : x1,
                "w1" : w1,
            }
        else:
            feature_data = None

        # Train model
        logger.info("Training model")
        trainer = RatioTrainer(self.model, n_workers=n_workers)
        result = trainer.train(
            data=data,
            data_val=data_val,
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
            feature_data = feature_data,
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
