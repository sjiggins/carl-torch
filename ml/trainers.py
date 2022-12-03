from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
from collections import OrderedDict
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score
logger = logging.getLogger(__name__)

class NanException(Exception):
    pass



class NumpyDataset(Dataset):
    """ Dataset for numpy arrays with explicit memmap support """

    def __init__(self, *arrays, **kwargs):

        self.dtype = kwargs.get("dtype", torch.float)
        self.memmap = []
        self.data = []
        self.n = None

        for array in arrays:
            if self.n is None:
                self.n = array.shape[0]
            assert array.shape[0] == self.n

            if isinstance(array, np.memmap):
                self.memmap.append(True)
                self.data.append(array)
            else:
                self.memmap.append(False)
                tensor = torch.from_numpy(array).to(self.dtype)
                self.data.append(tensor)

    def __getitem__(self, index):
        items = []
        for memmap, array in zip(self.memmap, self.data):
            if memmap:
                tensor = np.array(array[index])
                items.append(torch.from_numpy(tensor).to(self.dtype))
            else:
                items.append(array[index])
        return tuple(items)

    def __len__(self):
        return self.n


class Trainer(object):
    """ Trainer class. Any subclass has to implement the forward_pass() function. """

    def __init__(self, model, run_on_gpu=True, double_precision=False, n_workers=8):
        self._init_timer()
        self._timer(start="ALL")
        self._timer(start="initialize model")
        self.model = model
        self.run_on_gpu = run_on_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.run_on_gpu else "cpu")
        self.dtype = torch.double if double_precision else torch.float
        self.n_workers = n_workers
        self.model = self.model.to(self.device, self.dtype)

        logger.info(
            "Training on %s with %s precision",
            "GPU" if self.run_on_gpu else "CPU",
            "double" if double_precision else "single",
        )
        logger.info(" run_on_gpu %r,   torch.cuda.is_available() %r ", run_on_gpu, torch.cuda.is_available())

        self._timer(stop="initialize model")
        self._timer(stop="ALL")

    def train(
        self,
        data,
        loss_functions,
        loss_weights=None,
        loss_labels=None,
        epochs=50,
        batch_size=1,
        optimizer="lbfgs",
        optimizer_kwargs=None,
        initial_lr=0.001,
        final_lr=0.0001,
        data_val=None,
        validation_split=0.25,
        early_stopping=True,
        early_stopping_patience=None,
        clip_gradient=None,
        verbose="some",
        intermediate_train_plot=None,
        intermediate_save=None,
    ):
        self._timer(start="ALL")
        self._timer(start="check data")

        logger.debug("Initialising training data")
        self.check_data(data)
        self._timer(stop="check data", start="make dataset")
        print("Train Data")
        data_labels, dataset = self.make_dataset(data)
        if data_val is not None:
            print("Val. Data")
            _, dataset_val = self.make_dataset(data_val)
        else:
            dataset_val = None
        self._timer(stop="make dataset", start="make dataloader")
        train_loader, val_loader = self.make_dataloaders(dataset, dataset_val, validation_split, batch_size)

        self._timer(stop="make dataloader", start="setup optimizer")
        logger.debug("Setting up optimizer")
        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        opt = optimizer(self.model.parameters(), lr=initial_lr, **optimizer_kwargs)
        early_stopping = early_stopping and (validation_split is not None) and (epochs > 1)
        best_loss, best_model, best_epoch = None, None, None
        if early_stopping and early_stopping_patience is None:
            logger.debug("Using early stopping with infinite patience")
        elif early_stopping:
            logger.debug("Using early stopping with patience %s", early_stopping_patience)
        else:
            logger.debug("No early stopping")

        self._timer(stop="setup optimizer", start="initialize training")
        n_losses = len(loss_functions)
        loss_weights = [1.0] * n_losses if loss_weights is None else loss_weights

        # Verbosity
        if verbose == "all":  # Print output after every epoch
            n_epochs_verbose = 1
        elif verbose == "many":  # Print output after 2%, 4%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 50, 0)), 1)
        elif verbose == "some":  # Print output after 10%, 20%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 20, 0)), 1)
        elif verbose == "few":  # Print output after 20%, 40%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 5, 0)), 1)
        elif verbose == "none":  # Never print output
            n_epochs_verbose = epochs + 2
        else:
            raise ValueError("Unknown value %s for keyword verbose", verbose)
        logger.debug("Will print training progress every %s epochs", n_epochs_verbose)

        logger.debug("Beginning main training loop")
        losses_train, losses_val, accuracy_train, accuracy_val = [], [], [], []
        self._timer(stop="initialize training")

        # Loop over epochs
        for i_epoch in range(epochs):
            logger.debug("Training epoch", i_epoch + 1, epochs)
            self._timer(start="set lr")
            lr = self.calculate_lr(i_epoch, epochs, initial_lr, final_lr)
            self.set_lr(opt, lr)
            logger.debug("Learning rate: %s", lr)
            self._timer(stop="set lr")
            loss_val = None

            try:
                loss_train, loss_val, loss_contributions_train, loss_contributions_val, accu_train, accu_val = self.epoch(
                    i_epoch, data_labels, train_loader, val_loader, opt, loss_functions, loss_weights, clip_gradient
                )
                losses_train.append(loss_train)
                losses_val.append(loss_val)
                accuracy_train.append(accu_train)
                accuracy_val.append(accu_val)
            except NanException:
                logger.info("Ending training during epoch %s because NaNs appeared", i_epoch + 1)
                break

            self._timer(start="early stopping")
            if early_stopping:
                try:
                    best_loss, best_model, best_epoch = self.check_early_stopping(
                        best_loss, best_model, best_epoch, loss_val, i_epoch, early_stopping_patience
                    )
                except EarlyStoppingException:
                    logger.info("Early stopping: ending training after %s epochs", i_epoch + 1)
                    break
            self._timer(stop="early stopping", start="report epoch")

            # display the first 10 epoch
            if i_epoch < 10:
                verbose_epoch = i_epoch
            else:
                verbose_epoch = (i_epoch + 1) % n_epochs_verbose == 0
            self.report_epoch(
                i_epoch,
                loss_labels,
                loss_train,
                loss_val,
                loss_contributions_train,
                loss_contributions_val,
                accu_train = accu_train,
                accu_val = accu_val,
                verbose=verbose_epoch,
            )
            self._timer(stop="report epoch")

            # do intermediate plotting and saving
            if verbose_epoch:
                if intermediate_train_plot:
                    self._timer(start="intermediate train plot")
                    m_evaluator, m_plotter = intermediate_train_plot
                    for type in ["train", "val"]:
                        m_r_hat, m_s_hat = m_evaluator[0](m_evaluator[1][type])
                        m_plotter[1][type].update({"ext_plot_path":f"epoch_plot_{i_epoch}_{type}"})
                        m_carl_w = 1.0/m_r_hat
                        m_plotter[1][type].update({"weights":m_carl_w})
                        m_plotter[1][type].update({"label":type})
                        m_plotter[0](**m_plotter[1][type])
                    self._timer(stop="intermediate train plot")
                if intermediate_save:
                    self._timer(start="intermediate save")
                    saver, save_args = intermediate_save
                    m_filename = save_args['filename']
                    new_fname = f"models/epoch_{i_epoch}/{m_filename}"
                    save_args.update({"filename":new_fname})
                    saver(**save_args)
                    save_args.update({"filename":m_filename})
                    np.save(f"{new_fname}_loss_train.py", np.array(losses_train))
                    np.save(f"{new_fname}_loss_val.py", np.array(losses_val))
                    np.save(f"{new_fname}_accu_train.py", np.array(accuracy_train))
                    np.save(f"{new_fname}_accu_val.py", np.array(accuracy_val))
                    self._timer(stop="intermediate save")

        self._timer(start="early stopping")
        if early_stopping and len(losses_val) > 0:
            self.wrap_up_early_stopping(best_model, loss_val, best_loss, best_epoch)
        self._timer(stop="early stopping")

        logger.debug("Training finished")

        self._timer(stop="ALL")
        self._report_timer()

        return np.array(losses_train), np.array(losses_val), np.array(accuracy_train), np.array(accuracy_val)

    @staticmethod
    def report_data(data):
        logger.debug("Training data:")
        for key, value in six.iteritems(data):
            if value is None:
                logger.debug("  %s: -", key)
            else:
                logger.debug(
                    "  %s: shape %s, first %s, mean %s, min %s, max %s",
                    key,
                    value.shape,
                    value[0],
                    np.mean(value, axis=0),
                    np.nanmin(value, axis=0),  # originally np.min()
                    np.nanmax(value, axis=0),  # originally np.max()
                )

    @staticmethod
    def check_data(data):
        pass

    def make_dataset(self, data):
        data_arrays = []
        data_labels = []
        for key, value in six.iteritems(data):
            data_labels.append(key)
            print(key)
            print(len(key))
            print(len(value))
            data_arrays.append(value)
        dataset = NumpyDataset(*data_arrays, dtype=self.dtype, run_on_gpu=self.run_on_gpu)
        return data_labels, dataset

    def make_dataloaders(self, dataset, dataset_val, validation_split, batch_size):
        if dataset_val is None and (validation_split is None or validation_split <= 0.0):
            train_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, pin_memory=self.run_on_gpu, num_workers=self.n_workers
            )
            val_loader = None

        elif dataset_val is not None:
            train_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, pin_memory=self.run_on_gpu, num_workers=self.n_workers
            )
            val_loader = DataLoader(
                dataset_val, batch_size=batch_size, shuffle=True, pin_memory=self.run_on_gpu, num_workers=self.n_workers
            )

        else:
            assert 0.0 < validation_split < 1.0, "Wrong validation split: {}".format(validation_split)

            n_samples = len(dataset)
            indices = list(range(n_samples))
            split = int(np.floor(validation_split * n_samples))
            np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(
                dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                pin_memory=self.run_on_gpu,
                num_workers=self.n_workers,
            )
            val_loader = DataLoader(
                dataset,
                sampler=val_sampler,
                batch_size=batch_size,
                pin_memory=self.run_on_gpu,
                num_workers=self.n_workers,
            )

        return train_loader, val_loader

    @staticmethod
    def calculate_lr(i_epoch, n_epochs, initial_lr, final_lr):
        if n_epochs == 1:
            return initial_lr
        return initial_lr * (final_lr / initial_lr) ** float(i_epoch / (n_epochs - 1.0))

    @staticmethod
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def epoch(
        self,
        i_epoch,
        data_labels,
        train_loader,
        val_loader,
        optimizer,
        loss_functions,
        loss_weights,
        clip_gradient=None,
    ):
        n_losses = len(loss_functions)

        self.model.train()
        loss_contributions_train = np.zeros(n_losses)
        loss_train = 0.0
        accu_train = 0.0
        self._timer(start="load training batch")
        for i_batch, batch_data in enumerate(train_loader):
            batch_data = OrderedDict(list(zip(data_labels, batch_data)))
            self._timer(stop="load training batch")
            batch_loss, batch_loss_contributions, accuracy = self.batch_train(
                batch_data, loss_functions, loss_weights, optimizer, clip_gradient
            )
            loss_train += batch_loss
            accu_train += accuracy
            for i, batch_loss_contribution in enumerate(batch_loss_contributions):
                loss_contributions_train[i] += batch_loss_contribution

            self.report_batch(i_epoch, i_batch, batch_loss)

            self._timer(start="load training batch")
        self._timer(stop="load training batch")

        loss_contributions_train /= len(train_loader)
        loss_train /= len(train_loader)
        accu_train /= len(train_loader)

        if val_loader is not None:
            self.model.eval()
            loss_contributions_val = np.zeros(n_losses)
            loss_val = 0.0
            accu_val = 0.0

            self._timer(start="load validation batch")
            for i_batch, batch_data in enumerate(val_loader):
                batch_data = OrderedDict(list(zip(data_labels, batch_data)))
                self._timer(stop="load validation batch")

                batch_loss, batch_loss_contributions, accuracy = self.batch_val(batch_data, loss_functions, loss_weights)
                loss_val += batch_loss
                accu_val += accuracy
                for i, batch_loss_contribution in enumerate(batch_loss_contributions):
                    loss_contributions_val[i] += batch_loss_contribution

                self._timer(start="load validation batch")
            self._timer(stop="load validation batch")

            loss_contributions_val /= len(val_loader)
            loss_val /= len(val_loader)
            accu_val /= len(val_loader)

        else:
            loss_contributions_val = None
            loss_val = None
            accu_val = None

        return loss_train, loss_val, loss_contributions_train, loss_contributions_val, accu_train, accu_val

    def batch_train(self, batch_data, loss_functions, loss_weights, optimizer, clip_gradient=None):
        self._timer(start="training forward pass")
        loss_contributions, accuracy = self.forward_pass(batch_data, loss_functions)
        self._timer(stop="training forward pass", start="training sum losses")
        loss = self.sum_losses(loss_contributions, loss_weights)
        self._timer(stop="training sum losses", start="optimizer step")

        self.optimizer_step(optimizer, loss, clip_gradient)
        self._timer(stop="optimizer step", start="training sum losses")

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        self._timer(stop="training sum losses")

        return loss, loss_contributions, accuracy

    def batch_val(self, batch_data, loss_functions, loss_weights):
        self._timer(start="validation forward pass")
        loss_contributions, accuracy = self.forward_pass(batch_data, loss_functions)
        self._timer(stop="validation forward pass", start="validation sum losses")
        loss = self.sum_losses(loss_contributions, loss_weights)

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        self._timer(stop="validation sum losses")
        return loss, loss_contributions, accuracy

    def forward_pass(self, batch_data, loss_functions):
        """
        Forward pass of the model. Needs to be implemented by any subclass.
        Parameters
        ----------
        batch_data : OrderedDict with str keys and Tensor values
            The data of the minibatch.
        loss_functions : list of function
            Loss functions.
        Returns
        -------
        losses : list of Tensor
            Losses as scalar pyTorch tensors.
        """
        raise NotImplementedError

    @staticmethod
    def sum_losses(contributions, weights):
        loss = weights[0] * contributions[0]
        for _w, _l in zip(weights[1:], contributions[1:]):
            loss = loss + _w * _l
        return loss

    def optimizer_step(self, optimizer, loss, clip_gradient):
        # Zero gradients (optimizer.zero_grad()), perform a backward pass (loss.backward()), and update the weights (optimizer.step()).
        self._timer(start="opt: zero grad")
        optimizer.zero_grad()
        self._timer(stop="opt: zero grad", start="opt: backward")
        loss.backward()
        self._timer(start="opt: clip grad norm", stop="opt: backward")
        if clip_gradient is not None:
            clip_grad_norm_(self.model.parameters(), clip_gradient)
        self._timer(stop="opt: clip grad norm", start="opt: step")
        optimizer.step()
        self._timer(stop="opt: step")

    def check_early_stopping(self, best_loss, best_model, best_epoch, loss, i_epoch, early_stopping_patience=None):
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_model = self.model.state_dict()
            best_epoch = i_epoch

        if early_stopping_patience is not None and i_epoch - best_epoch > early_stopping_patience >= 0:
            raise EarlyStoppingException

        if loss is None or not np.isfinite(loss):
            raise EarlyStoppingException

        return best_loss, best_model, best_epoch

    @staticmethod
    def report_batch(i_epoch, i_batch, loss_train):
        if i_batch in [0, 1, 10, 100, 1000]:
            logger.debug("  Epoch {:>3d}, batch {:>3d}: loss {:>8.5f}".format(i_epoch + 1, i_batch + 1, loss_train))

    @staticmethod
    def report_epoch(
        i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, accu_train=None, accu_val=None, verbose=False
    ):
        logging_fn = logger.info if verbose else logger.debug

        def contribution_summary(labels, contributions):
            summary = ""
            for i, (label, value) in enumerate(zip(labels, contributions)):
                if i > 0:
                    summary += ", "
                summary += "{}: {:>6.3f}".format(label, value)
            return summary

        train_report = "  Epoch {:>3d}: train loss {:>8.8f} ({}), accu {}".format(
            i_epoch + 1, loss_train, contribution_summary(loss_labels, loss_contributions_train), accu_train
        )
        logging_fn(train_report)

        if loss_val is not None:
            val_report = "             val. loss  {:>8.8f} ({}), accu {}".format(
                loss_val, contribution_summary(loss_labels, loss_contributions_val), accu_val
            )
            logging_fn(val_report)

    def wrap_up_early_stopping(self, best_model, currrent_loss, best_loss, best_epoch):
        if best_loss is None or not np.isfinite(best_loss):
            logger.warning("Best loss is None, cannot wrap up early stopping")
        elif currrent_loss is None or not np.isfinite(currrent_loss) or best_loss < currrent_loss:
            logger.info(
                "Early stopping after epoch %s, with loss %8.5f compared to final loss %8.5f",
                best_epoch + 1,
                best_loss,
                currrent_loss,
            )
            self.model.load_state_dict(best_model)
        else:
            logger.info("Early stopping did not improve performance")

    @staticmethod
    def _check_for_nans(label, *tensors):
        for tensor in tensors:
            if tensor is None:
                continue
            if torch.isnan(tensor).any():
                logger.warning("%s contains NaNs, aborting training!", label)
                raise NanException

    def _init_timer(self):
        self.timer = OrderedDict()
        self.time_started = OrderedDict()

    def _timer(self, start=None, stop=None):
        if start is not None:
            self.time_started[start] = time.time()

        if stop is not None:
            if stop not in list(self.time_started.keys()):
                logger.warning("Timer for task %s has been stopped without being started before", stop)
                return

            dt = time.time() - self.time_started[stop]
            del self.time_started[stop]

            if stop in list(self.timer.keys()):
                self.timer[stop] += dt
            else:
                self.timer[stop] = dt

    def _report_timer(self):
        logger.info("Training time spend on:")
        for key, value in six.iteritems(self.timer):
            logger.info("  {:>32s}: {:6.2f}h".format(key, value / 3600.0))



class RatioTrainer(Trainer):
    def __init__(self, model, run_on_gpu=True, double_precision=False, n_workers=8):
        super(RatioTrainer, self).__init__(model, run_on_gpu, double_precision, n_workers)

    def check_data(self, data):
        data_keys = list(data.keys())

    def forward_pass(self, batch_data, loss_functions):
        self._timer(start="fwd: move data")
        x = batch_data["x"].to(self.device, self.dtype, non_blocking=True)
        y = batch_data["y"].to(self.device, self.dtype, non_blocking=True)
        w = batch_data["w"].to(self.device, self.dtype, non_blocking=True) #sjiggins

        self._timer(stop="fwd: move data", start="fwd: check for nans")
        self._timer(start="fwd: model.forward", stop="fwd: check for nans")

        r_hat, s_hat= self.model(x)

        self._timer(stop="fwd: model.forward", start="fwd: check for nans")
        self._check_for_nans("Model output", s_hat, r_hat)

        self._timer(start="fwd: calculate losses", stop="fwd: check for nans")
        losses = [
            loss_function(s_hat, y, w) for loss_function in loss_functions
        ]

        # computing binary classification accuracy
        truth = (y>0.5).float()*1
        predict = (s_hat>0.0).float()*1
        accuracy = accuracy_score(truth.cpu().flatten(), predict.cpu().flatten(), sample_weight=w.cpu().flatten())

        self._timer(stop="fwd: calculate losses", start="fwd: check for nans")
        self._check_for_nans("Loss", *losses)
        self._timer(stop="fwd: check for nans")

        return losses, accuracy
