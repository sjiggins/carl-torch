import numpy as np

from itertools import product
from sklearn.utils import check_array
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class Histogram():
    """N-dimensional histogram."""

    def __init__(self, bins=10, range=None, interpolation=None,
                 variable_width=False):
        """Constructor.
        Parameters
        ----------
        * `bins` [int or string]:
            The number of bins or `"blocks"` for bayesian blocks.
        * `range` [list of bounds (low, high)]:
            The boundaries. If `None`, bounds are inferred from data.
        * `interpolation` [string, optional]
            Specifies the kind of interpolation between bins as a string
            (`"linear"`, `"nearest"`, `"zero"`, `"slinear"`, `"quadratic"`,
            `"cubic"`).
        * `variable_width` [boolean, optional]
            If True use equal probability variable length bins.
        """
        self.bins = bins
        self.range = range
        self.interpolation = interpolation
        self.variable_width = variable_width


    def pdf(self, X, **kwargs):
        X = check_array(X)
        if self.ndim_ == 1 and self.interpolation:
            return self.interpolation_(X[:, 0])

        all_indices = []

        for j in range(X.shape[1]):
            indices = np.searchsorted(self.edges_[j],
                                      X[:, j],
                                      side="right") - 1

            # For the last bin, the upper is inclusive
            indices[X[:, j] == self.edges_[j][-2]] -= 1
            all_indices.append(indices)
        return self.histogram_[tuple(all_indices)]

    def nll(self, X, **kwargs):
        return -np.log(self.pdf(X, **kwargs))

    def fit(self, X, sample_weight=None, **kwargs):
        # Checks
        X = check_array(X)
        if sample_weight is not None and len(sample_weight) != len(X):
            raise ValueError
        # Compute histogram and edges
        if self.bins == "blocks":
            bins = bayesian_blocks(X.ravel(), fitness="events", p0=0.0001)
            range_ = self.range[0] if self.range else None
            h, e = np.histogram(X.ravel(), bins=bins, range=range_,
                                weights=sample_weight, normed=False)
            e = [e]

        elif self.variable_width:
            ticks = [np.percentile(X.ravel(), 100. * k / self.bins) for k
                     in range(self.bins + 1)]
            ticks[-1] += 1e-5
            range_ = self.range[0] if self.range else None
            h, e = np.histogram(X.ravel(), bins=ticks, range=range_,
                                normed=False, weights=sample_weight)
            h, e = h.astype(float), e.astype(float)
            widths = e[1:] - e[:-1]
            h = h / widths / h.sum()
            e = [e]

        else:
            bins = self.bins
            h, e = np.histogramdd(X, bins=bins, range=self.range,
                                  weights=sample_weight.flatten(), normed=True)

        # Plot histogram for prosperity
        pltx = []
        for i in range(bins):
            pltx.append( (e[0][i+1] + e[0][i]) /2 )
        plt.plot(pltx,h)
        plt.xlabel('Prob(y=1 | X)')
        plt.ylabel('Entries')
        plt.savefig("plots/"+kwargs["global_name"]+"/"+kwargs["output"]+'.png')
        plt.clf()

        # Add empty bins for out of bound samples
        for j in range(X.shape[1]):
            h = np.insert(h, 0, 0., axis=j)
            h = np.insert(h, h.shape[j], 0., axis=j)
            e[j] = np.insert(e[j], 0, -np.inf)
            #e[j] = np.insert(e[j], 0, 0.0)
            e[j] = np.insert(e[j], len(e[j]), np.inf)
            #e[j] = np.insert(e[j], len(e[j]), 1.0)

        if X.shape[1] == 1 and self.interpolation:
            inputs = e[0][2:-1] - (e[0][2] - e[0][1]) / 2.
            inputs[0] = e[0][1]
            inputs[-1] = e[0][-2]
            outputs = h[1:-1]
            self.interpolation_ = interp1d(inputs, outputs,
                                           kind=self.interpolation,
                                           bounds_error=False,
                                           fill_value=0.)
        self.histogram_ = h
        self.edges_ = e
        self.ndim_ = X.shape[1]
        return self

    @property
    def ndim(self):
        return self.ndim_
