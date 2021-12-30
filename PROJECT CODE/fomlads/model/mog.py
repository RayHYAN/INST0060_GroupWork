import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal

from fomlads.data.wrangle import subsample_datapoints


def sample_mog(N, means, covmtxs, mixcoefs):
    """
    Sample 2d data from a mixture of gaussians distribution

    Parameters
    ----------
    N - the number of data points to output
    means - a KxD array of mean vectors (one per row)
    covmtxs - a KxDxD array of covariance matrices each row slice covmtxs[k,:,:]
        is a DxD covariance matrix 
    mixcoefs - a 1d array (length K) of mixture coefficients (must sum to 1)
    

    Returns
    -------
    datamtx - randomly sampled data NxD array
    latent_components - a vector of length N, specifying which component 
        generated the data (as an int)
    """
    if not math.isclose(np.sum(mixcoefs), 1.):
        # checks if the sum of the mixture coefficients is very close to 1.
        # it should be (precision errors sometimes mean we do not get exactly 1)
        raise ValueError("Mixture coefficients do not sum to  1")
    K, D = means.shape
    # choose which component each sample comes from
    latent_components = np.random.choice(
        np.arange(K), N, replace=True, p=mixcoefs)
    datamtx = np.empty((N,D))
    for k in range(K):
        mean = means[k,:]
        covmtx = covmtxs[k,:,:]
        # identify which points are formed from this component
        component_points = (latent_components == k)
        Nk = np.sum(component_points)
        # generate points from this component
        datamtx[component_points] = \
            np.random.multivariate_normal(mean, covmtx, size=Nk)
    # return data matrix and latent components
    return datamtx, latent_components

def em_mog(
        datamtx, K, initial_means=None, initial_covmtxs=None,
        initial_mixcoefs=None, threshold=1e-10, iterations=None):
    """
    The Expectation maximisation algorithm for mixture of Gaussians

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)
    K - integer number of components to fit
    initial_means (optional) - KxD matrix of initial component means
    initial_covmtxs (optional) - KxDxD matrix of initial component covariances
    initial_mixcoefs (optional) - K vector of initial mixture coefficients
    threshold (optional) - the threshold magnitude for termination
    iterations (optional) - the maximum number of iterations

    returns
    -------
    means - KxD matrix of component means
    covmtxs - KxDxD matrix of component covariances
    mixcoefs - K vector of mixture coefficients
    responsibilties - NxK matrix of soft-assignments, one row per datapoint, one
        column per component
    """
    N, D = datamtx.shape
    # initialise the cluster parameters
    if initial_means is None:
        # default means are sampled from the data
        means = subsample_datapoints(datamtx, K)
    else:
        means = initial_means
        assert means.shape == (K,D), "Initial means are incompatible"
    if initial_covmtxs is None:
        # default covariances are the identity matrix
        covmtxs = np.empty((K,D,D))
        for k in range(K):
            covmtxs[k,:,:] = np.identity(D)
    else:
        covmtxs = initial_covmtxs
        assert covmtxs.shape == (K,D,D), "Initial covariances are incompatible"
    if initial_mixcoefs is None:
        # default mixture coefficients is all equal
        mixcoefs = np.ones(K)/K
    else:
        covmtxs = initial_covmtxs
        assert covmtxs.shape == (K,D,D), "Initial covariances are incompatible"
    # initially change must be larger than threshold for algorithm to run 
    # at least one iteration
    change = 2*threshold
    iteration = 1
    while change > threshold:
        log_resps = em_e_step(datamtx, means, covmtxs, mixcoefs)
        old_means = means
        old_covmtxs = covmtxs
        old_mixcoefs = mixcoefs
        means, covmtxs, mixcoefs = em_m_step(datamtx, log_resps)
        # the change is the sum of squared differences across all parameters 
        change = np.sum((means-old_means)**2) \
            + np.sum((covmtxs-old_covmtxs)**2) \
            + np.sum((mixcoefs-old_mixcoefs)**2)
        if not iterations is None and iteration >= iterations:
            break
        iteration += 1
    #
    return means, covmtxs, mixcoefs, log_resps

def em_e_step(datamtx, means, covmtxs, mixcoefs):
    """
    The E step for the EM algorithm -- mixture of Gaussians

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)
    means - (KxD) matrix of component mean estimates
    covmtxs - (KxDxD) matrix of component covariance estimates
    mixcoefs - K vector of mixture coefficient estimates
  
    returns
    -------
    log_resps - an NxK matrix of logs of responsibility vectors, namely
        logs of soft-assignments of each data-point (row) to each component
    """
    N,D = datamtx.shape
    K = means.shape[0]
    # initially we construct the log_resps in unnormalised way then
    # later we will normalise by ensuring each row sums to 1
    log_resps = np.empty((N,K))
    for k in range(K):
        mean = means[k,:]
        covmtx = covmtxs[k,:,:]
        mixcoef = mixcoefs[k]
        # calculate N(x|mu_k, Sigma_k) for each x
        kth_component_logdensities = multivariate_normal.logpdf(datamtx, mean, covmtx)
        # now insert pi_k *N(x|mu_k, Sigma_k) as kth column of matrix
        log_resps[:,k] = np.log(mixcoef) + kth_component_logdensities
    # now renormalise the rows (so exponential of each row sums to 1)
    # We work in logs because they are less sensitive to precision errors
    normaliser = np.log(np.exp(log_resps).sum(axis=1))
    log_resps = log_resps - normaliser.reshape((N,1))
    return log_resps

def em_m_step(datamtx, log_resps):
    """
    The M step for the EM algorithm -- mixture of Gaussians

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)
    log_resps - an NxK matrix of logs of responsibility vectors, i.e. logs of
        soft-assignments of each data-point (row) to each component
  
    returns
    -------
    means - (KxD) matrix of component new mean estimates
    covmtxs - (KxDxD) matrix of component new covariance estimates
    mixcoefs - K vector of new mixture coefficient estimates
    """
    N, D = datamtx.shape
    K = log_resps.shape[1]
    # calculate the estimated number of points assigned to each component
    Nks = np.exp(log_resps).sum(axis=0)
    mixcoefs = Nks/N
    means = np.empty((K,D))
    covmtxs = np.empty((K, D, D))
    for k, Nk in enumerate(Nks):
        # resps_k - shorthand for the responsibilities rnk) for all n
        resps_k = np.exp(log_resps[:,k])
        meank = np.sum(datamtx*resps_k.reshape((N,1)), axis=0)/Nk
        means[k,:] = meank
        # construct a sequence of vectors (x_n - mu_k) as matrix objects
        diffs_k = ( np.matrix(xn - meank).reshape((D,1)) for xn in datamtx )
        # a sequence of component matrices each a term from sum of Result (7.6)
        covmtx_components = (
            rnk*diff*diff.T for rnk, diff in zip(resps_k, diffs_k))
        # performs sum from Result (7.6) giving covariance for component k
        covmtxs[k, :, :] = np.sum(covmtx_components, axis=0)/Nks[k]
    return means, covmtxs, mixcoefs

def log_likelihood_mog(datamtx, means, covmtxs, mixcoefs):
    """
    Calculate the total joint likelihood of the data given the model parameters

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)
    means - (KxD) matrix of component mean estimates
    covmtxs - (KxDxD) matrix of component covariance estimates
    mixcoefs - K vector of mixture coefficient estimates
  
    returns
    -------
    log_likelihood - a scalar representing the joint log likelihood of the data
        given the model parameters
        
        each datapoint xn:   p(xn) = \sum_k \pi_k N(xn|meank, covmtxk)
    """
    N, D = datamtx.shape
    K = means.shape[0]
    # the vector probs will ultimately store the probability density for each
    # data point, i.e. probs[n] := p(x_n)
    probs = np.zeros(N)
    for k in range(K):
        mixcoefk = mixcoefs[k]
        meank = means[k,:]
        covmtxk = covmtxs[k,:,:]
        probs += mixcoefk * multivariate_normal.pdf(
            datamtx, meank, covmtxk)
    return np.sum(np.log(probs))

def aic_mog(datamtx, means, covmtxs, mixcoefs):
    """
    Calculates the AIC (the Akaike's information criterion) for the mixture of
    Gaussian's model.

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)
    means - (KxD) matrix of component mean estimates
    covmtxs - (KxDxD) matrix of component covariance estimates
    mixcoefs - K vector of mixture coefficient estimates
  
    returns
    -------
    aic - Akaike's information criterion
    """
    N, D = datamtx.shape
    K = means.shape[0]
    log_L = log_likelihood_mog(datamtx, means, covmtxs, mixcoefs)
    num_params = K*(D**2 + D)/2. + K - 1
    return 2* num_params - 2 *log_L

def bic_mog(datamtx, means, covmtxs, mixcoefs):
    """
    Calculates the BIC (the Bayesian information criterion) for the mixture of
    Gaussian's model.

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)
    means - (KxD) matrix of component mean estimates
    covmtxs - (KxDxD) matrix of component covariance estimates
    mixcoefs - K vector of mixture coefficient estimates
  
    returns
    -------
    bic - Bayesian information criterion
    """
    N, D = datamtx.shape
    K = means.shape[0]
    log_L = log_likelihood_mog(datamtx, means, covmtxs, mixcoefs)
    num_params = K*(D**2 + D)/2. + K - 1
    return num_params*(np.log(N) - np.log(2*np.pi)) - 2 *log_L


