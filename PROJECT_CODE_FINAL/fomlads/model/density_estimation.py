import numpy as np

def max_lik_1d_gaussian(samples):
    """
    parameters
    ----------
      samples - a 1d array of samples
    returns
    -------
      mu_ml - a maximum likelihood estimate of the mean
      sigma2_ml - a maximum likelihood estimate of the variance
    """
    N = samples.size
    mu_ml = np.sum(samples)/N
    sigma2_ml = np.sum((samples-mu_ml)**2)/N
    return mu_ml, sigma2_ml


def posterior_1d_gaussian(N, m_0, s2_0, mu_ml, sigma2):
    """
    parameters
    ----------
      N - number of samples
      m_0 - prior mean for data mean mu
      s2_0 - prior variance for data mean mu
      mu_ml - maximum likelihood mean
      sigma2 - known data variance

    returns
    -------
      m_N - posterior mean for data mean mu
      s2_N - posterior variance for data mean mu
    """
    m_N = (sigma2*m_0 + N*s2_0*mu_ml)/(N*s2_0 + sigma2) 
    s2_N = 1/(1/s2_0 + N/sigma2)
    return m_N, s2_N

def max_lik_mv_gaussian(data):
    """
    Finds the maximum likelihood mean and covariance matrix for gaussian data
    samples (data)

    parameters
    ----------
    data - data array, 2d array of samples, each row is assumed to be an
      independent sample from a multi-variate gaussian

    returns
    -------
    mu - mean vector
    Sigma - 2d array corresponding to the covariance matrix  
    """
    # the mean sample is the mean of the rows of data
    N, dim = data.shape
    mu = np.mean(data,0)
    Sigma = np.zeros((dim,dim))
    # the covariance matrix requires us to sum the dyadic product of
    # each sample minus the mean.
    for x in data:
        # subtract mean from data point, and reshape to column vector
        # note that numpy.matrix is being used so that the * operator
        # in the next line performs the outer-product v * v.T 
        x_minus_mu = (x - mu).reshape((dim,1))
        # the outer-product v @ v.T of a k-dimentional vector v gives
        # a (k x k)-matrix as output. This is added to the running total.
        Sigma += x_minus_mu @ x_minus_mu.T
    # Sigma is unnormalised, so we divide by the number of datapoints
    Sigma /= N
    return mu, Sigma


def max_lik_mv_gaussian_fast(X):
    """
    A faster alternative to the max_lik_mv_gaussian function.

    Finds the maximum likelihood mean and variance for gaussian data samples (X)

    parameters
    ----------
    X - data array, 2d array of samples, each row is assumed to be an
      independent sample from a multi-variate gaussian

    returns
    -------
    mu - mean vector
    Sigma - 2d array corresponding to the covariance matrix  
    """
    # the mean sample is the mean of the rows of X
    N, dim = X.shape
    mu = np.mean(X,axis=0)
    # the covariance matrix is requires us to sum the dyadic product of
    # each sample minus the mean. This can be done compactly like this:
    # This line produces a sequence of column vectors (x-mu) one for each x
    x_minus_mus = ((x - mu).reshape((dim,1)) for x in X)
    # This line takes the mean over the dyadic products (x-mu)(x-mu).T of all x
    Sigma = np.sum(x_minus_mu @ x_minus_mu.T for x_minus_mu in x_minus_mus)/N
    # we convert Sigma matrix back to an array to avoid confusion later
    return mu, Sigma

