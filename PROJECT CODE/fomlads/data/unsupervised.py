import numpy as np
import numpy.random as random
import math 

from fomlads.model.mog import sample_mog

def scenario1_data(N):
    # assign the true means
    true_means = np.empty((3,2))
    true_means[0,:] = [-5, -1]
    true_means[1,:] = [0, 1]
    true_means[2,:] = [5, -1]
    # assign the covariances (each the identity matrix)
    covmtxs = np.empty((3,2,2))
    covmtxs[0,:,:] = np.identity(2)
    covmtxs[1,:,:] = np.identity(2)
    covmtxs[2,:,:] = np.identity(2)
    # and the mixture coefficients
    mixcoefs = np.array([0.5, 0.3, 0.2])

    datamtx, latent_components = sample_mog(
        N, true_means, covmtxs, mixcoefs)
    return datamtx, latent_components, true_means, covmtxs, mixcoefs


def scenario2_data(N, K = 8):
    # assign the true means
    true_means = np.random.uniform(-10,10,(K,2))
    # assign the covariances (each diagonal with some variation in scales)
    covmtxs = np.zeros((K,2,2))
    covmtxs[:,0,0] = 2**np.random.uniform(-0.5,0.5, K)
    covmtxs[:,1,1] = 2**np.random.uniform(-0.5,0.5, K)
    # and the mixture coefficients all equal
    mixcoefs = np.ones(K)/K

    datamtx, latent_components = sample_mog(
        N, true_means, covmtxs, mixcoefs)
    return datamtx, latent_components, true_means, covmtxs, mixcoefs

def scenario3_data(N):
    # assign the true means
    K = 3
    true_means = np.empty((3,2))
    sep = 2.8
    corr = 0.8
    true_means[0,:] = [-sep, -0.5]
    true_means[1,:] = [0, 0]
    true_means[2,:] = [sep, 0.5]
    # assign the covariances (each the identity matrix)
    covmtxs = np.empty((3,2,2))
    covmtxs[0,:,:] = np.array([[1,corr],[corr,1]])
    covmtxs[1,:,:] = np.array([[1,-corr],[-corr,1]])
    covmtxs[2,:,:] = np.array([[1,corr],[corr,1]])
    # and the mixture coefficients
    mixcoefs = np.array([0.5, 0.3, 0.2])

    datamtx, latent_components = sample_mog(
        N, true_means, covmtxs, mixcoefs)
    return datamtx, latent_components, true_means, covmtxs, mixcoefs



def scenario4_data(N):
    # assign the true means
    datamtx = np.random.uniform(-10,10,(N,2))
    # there are no latent components and no mixture of gaussians components
    return datamtx
