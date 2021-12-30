import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def quadratic_feature_mapping(datamtx):
    """
    parameters
    ----------
    datamtx - the input data matrix

    returns
    -------
    designmtx- the design matrix
    """
    #  to enable python's broadcasting capability we need the datamtx array
    # as a NxDx1 array
    if len(datamtx.shape) == 1:
        # if the datamtx is just an array of scalars, turn this into
        # a Nx1x1 array
        datamtx = datamtx.reshape((datamtx.size,1))
    N, D = datamtx.shape
    M = 1 + D + D*(D+1)//2
    designmtx = np.empty((N,M))
    designmtx = np.hstack((np.ones((N,1)), datamtx))
    for e in range(D):
        localmtx = datamtx[:,e].reshape((N,1)) * datamtx[:,e:]
        designmtx = np.hstack((designmtx, localmtx))
    return designmtx

def construct_rbf_feature_mapping(centres, scale, constant_column=False):
    """
    parameters
    ----------
    centres - a DxM matrix (numpy array) where D is the dimension of the space
        and each row is the central position of an rbf basis function.
        For D=1 can pass an M-vector (numpy array).
    scale - a float determining the width of the distribution. Equivalent role
        to the standard deviation in the Gaussian distribution.

    returns
    -------
    feature_mapping - a function which takes an NxD data matrix and returns
        the design matrix (NxM matrix of features)
    """
    #  to enable python's broadcasting capability we need the centres
    # array as a 1xDxM array
    if len(centres.shape) == 1:
        centres = centres.reshape((1,1,centres.size))
    else:
        centres = centres.T.reshape((1,centres.shape[1],centres.shape[0]))
    # the denominator
    denom = 2*scale**2
    # now create a function based on these basis functions
    def feature_mapping(datamtx):
        #  to enable python's broadcasting capability we need the datamtx array
        # as a NxDx1 array
        if len(datamtx.shape) == 1:
            # if the datamtx is just an array of scalars, turn this into
            # a Nx1x1 array
            datamtx = datamtx.reshape((datamtx.size,1,1))
        else:
            # if datamtx is NxD array, then we reshape matrix as a
            # NxDx1 array
            datamtx = datamtx.reshape((datamtx.shape[0], datamtx.shape[1], 1))
        designmtx = np.exp(-np.sum((datamtx - centres)**2,1)/denom)
        if constant_column:
          designmtx = np.hstack((np.ones((designmtx.shape[0],1)), designmtx))
        return designmtx
    # return the created function
    return feature_mapping

# this is the feature mapping for a polynomial of given degree in 1d
def expand_to_monomials(inputs, degree):
    """
    Create a design matrix from a 1d array of input values, where columns
    of the output are powers of the inputs from 0 to degree (inclusive)

    So if input is: inputs=np.array([x1, x2, x3])  and degree = 4 then
    output will be design matrix:
        np.array( [[  1.    x1**1   x1**2   x1**3   x1**4   ]
                   [  1.    x2**1   x2**2   x2**3   x2**4   ]
                   [  1.    x3**1   x3**2   x3**3   x3**4   ]])
    """
    expanded_inputs = []
    for i in range(degree+1):
        expanded_inputs.append(inputs**i)
    return np.array(expanded_inputs).transpose()

