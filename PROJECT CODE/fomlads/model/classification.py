import csv
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from fomlads.data.function import logistic_sigmoid
from fomlads.model.density_estimation import max_lik_mv_gaussian

def project_data(data, weights):
    """
    Projects data onto single dimension according to some weight vector

    parameters
    ----------
    data - a 2d data matrix (shape NxD array-like)
    weights -- a 1d weight vector (shape D array like)

    returns
    -------
    projected_data -- 1d vector (shape N np.array)
    """
    N, D = data.shape
    weights = weights.reshape((D,1))
    projected_data = data @ weights
    return projected_data.flatten()

def fisher_linear_discriminant_projection(inputs, targets):
    """
    Finds the direction of best projection based on Fisher's linear discriminant

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1

    returns
    -------
    weights - a normalised projection vector corresponding to Fisher's linear 
        discriminant
    """
    # get the shape of the data
    N, D = inputs.shape
    # separate the classes
    inputs0 = inputs[targets==0]
    inputs1 = inputs[targets==1]
    N0 = inputs0.shape[0]
    N1 = inputs1.shape[0]
    # find maximum likelihood approximations to the two data-sets
    m0, S_0 = max_lik_mv_gaussian(inputs0)
    m1, S_1 = max_lik_mv_gaussian(inputs1)
    # convert the mean vectors to column vectors
    m0 = m0.reshape((D,1))
    m1 = m1.reshape((D,1))
    # calculate the total within-class covariance matrix
    S_W = (N0*S_0 + N1*S_1)
    # calculate weights vector
    weights = np.linalg.inv(S_W) @ (m1-m0)
    # normalise
    weights = weights/np.sum(weights)
    # we want to make sure that the projection is in the right direction
    # i.e. giving larger projected values to class1 so:
    projected_m0 = np.mean(project_data(inputs0, weights))
    projected_m1 = np.mean(project_data(inputs1, weights))
    if projected_m0 > projected_m1:
        weights = -weights
    return weights

def maximum_separation_projection(inputs, targets):
    """
    Finds the projection vector that maximises the distance between the 
    projected means

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1

    returns
    -------
    weights - a normalised projection vector
    """
    # get the shape of the data
    N, D = inputs.shape
    # separate the classes
    inputs0 = inputs[targets==0]
    inputs1 = inputs[targets==1]
    # find maximum likelihood approximations to the two data-sets
    m0,_ = max_lik_mv_gaussian(inputs0)
    m1,_ = max_lik_mv_gaussian(inputs1)
    # calculate weights vector
    weights = m1-m0
    return weights

