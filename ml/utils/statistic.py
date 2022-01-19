import numpy as np
from scipy.stats import wasserstein_distance

def compute_probs(data, weights, n=100):
    h, e = np.histogram(data, weights=weights, bins=n)
    p = h/np.sum(weights) #data.shape[0]
    return e, p

def support_intersection(p, q):
    sup_int = (
        list(
            filter(
                lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
            )
        )
    )
    return sup_int

def get_probs(list_of_tuples):
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

def kl_divergence(p, q):
    log_term = p*np.log(p/q)
    log_term = np.nan_to_num(log_term, nan=0.0, posinf = 0.0, neginf=0.0)
    return np.sum(log_term)

def compute_kl_divergence(train_sample, train_weights, test_sample, test_weights, n_bins=10):
    """
    Computes the KL Divergence using the support
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, train_weights, n=n_bins)
    _, q = compute_probs(test_sample, test_weights, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)

    return kl_divergence(p, q)

def wasserstein(x0, w0, x1, w1):
    """
    all the weights have to be non-negative
    """
    return wasserstein_distance(x0, x1, np.abs(w0), np.abs(w1))
