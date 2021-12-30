import numpy as np
import numpy.random as random

def sample_1d_regression_data(
        N, true_func=None, bind_limits=True, noise=0.3, seed=None,
        xlim=None):
    """
    Sample 1d input and target data for regression. Produces random inputs
    between 0 and 1 (default) and noise corrupted outputs of the true function.

    Parameters
    ----------
    N - the number of data points to output
    true_func - the true underlying function 
    bind_limits (optional) - whether to include points on the upper and lower
      limits in the inputs (this can improve the stability of your regression
      results).
    seed (optional) - the seed value (integer) for the pseudo-random number
        generator, allows one to recreate experiments
    xlim (optional) - the limits of the range of the input data

    Returns
    -------
    inputs - randomly sampled input data (x)
    targets - the associated targets (true_func(x) + gaussian noise)
    """
    if xlim is None:
        low, high = (0, 1)
    else:
        low, high = xlim
    if not seed is None:
        np.random.seed(seed)
    # if no underlying function is specified use the first arbitrary function
    # provided above
    if true_func is None:
        true_func = arbitrary_function_1
    # inputs are a collection of N random numbers between 0 and 1
    # for stability we include points at 0 and 1.
    inputs = random.uniform(low=low, high=high, size=N)
    if bind_limits and N >2:
        inputs[0] = low
        inputs[-1] = high
    # outputs are sin(2*pi*x) + gaussian noise
    targets = true_func(inputs) + random.normal(loc=0.0, scale=noise, size=N)
    return inputs, targets


def sample_2d_isotropic_gaussian(num_points, mux=3, sigma2x=25, muy=0, sigma2y=4):
    """
    Creates a (num_points x 2) array of random variables. Each column
    is a vector of values independently sampled from a normal distribution.
    """
    xsamples = np.random.normal(mux,np.sqrt(sigma2x),(num_points,1))
    ysamples = np.random.normal(muy,np.sqrt(sigma2y),(num_points,1))
    synthetic_data = np.hstack((xsamples, ysamples))
    return synthetic_data


