import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

from fomlads.model.basis_functions import expand_to_monomials

def ml_weights(inputs, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """
    # renaming for readability
    Phi = inputs
    targets = targets.reshape((-1,1))
    weights = linalg.inv(Phi.T @ Phi) @ Phi.T @ targets
    return weights.flatten()

# function goes by two names
least_squares_weights = ml_weights

def regularised_ml_weights(
        inputs, targets, lambda_):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets penalised by some regularisation term
    (lambda_)
    """
    # renaming for readability
    Phi = inputs
    targets = targets.reshape((-1,1))
    I = np.identity(Phi.shape[1])
    weights = linalg.inv(lambda_*I + Phi.T @ Phi) @ Phi.T @ targets
    return weights.flatten()

# function goes by two names
regularised_least_squares_weights = regularised_ml_weights

def linear_model_predict(inputs, weights):
    weights = weights.reshape((-1,1))
    ys = inputs@ weights
    return ys.flatten()

def calculate_weights_posterior(inputs, targets, beta, m0, S0):
    """
    Calculates the posterior distribution (multivariate gaussian) for weights
    in a linear model.

    parameters
    ----------
    inputs - 2d (N x K) array of inputs (data-matrix or design-matrix) where
        N is the number of data-points and each row is that point's
        representation
    targets - 1d (N)-array of target values
    beta - the known noise precision
    m0 - prior mean (vector) 1d-array (or array-like) of length K
    S0 - the prior covariance matrix 2d-array

    returns
    -------
    mN - the posterior mean (1d array)
    SN - the posterior covariance matrix 
    """
    N, K = inputs.shape
    targets = targets.reshape((N,1))
    m0 = m0.reshape((K,1))
    S0_inv = np.linalg.inv(S0)
    SN = np.linalg.inv(S0_inv + beta*inputs.T @ inputs)
    mN = SN @ (S0_inv @ m0 + beta*inputs.T @ targets )
    return mN.flatten(), SN

def predictive_distribution(inputs, beta, mN, SN):
    """
    Calculates the predictive distribution a linear model. This amounts to a
    mean and variance for each input point.

    parameters
    ----------
    inputs - 2d (N x K) array of inputs (data-matrix or design-matrix) where
        N is the number of data-points and each row is that point's
        representation
    beta - the known noise precision
    mN - posterior mean of the weights (vector) 1d-array (or array-like)
        of length K
    SN - the posterior covariance matrix for the weights 2d (K x K)-array 

    returns
    -------
    ys - a vector of mean predictions, one for each input datapoint
    sigma2Ns - a vector of variances, one for each input data-point 
    """
    N, K = inputs.shape
    mN = mN.reshape((K,1))
    ys = inputs @ mN
    # create an array of the right size with the uniform term
    sigma2Ns = np.ones(N)/beta
    for n in range(N):
        # now calculate and add in the data dependent term
        # NOTE: I couldn't work out a neat way of doing this without a for-loop
        # NOTE: but if anyone can, then please share the answer.
        phi_n = inputs[n,:].T
        sigma2Ns[n] += phi_n.T @ SN @ phi_n
    return ys.flatten(), sigma2Ns

def construct_polynomial(degree, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights. It returns a polynomial of the univariate input x
    # if given an array of xs, then it gives one functional value per input
    def polynomial_function(xs):
        monomials_mtx = expand_to_monomials(xs, degree)
        ys = monomials_mtx @ weights.reshape((-1,1))
        return ys.flatten()
    # we return the function reference (handle) itself. This can be used like
    # any other function
    return polynomial_function

def construct_feature_mapping_function(
        feature_mapping, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
        designmtx = feature_mapping(xs)
        return linear_model_predict(designmtx, weights)
    # we return the function itself as a variable. This can be used like
    # any other function
    return prediction_function

def construct_knn_function_1d(training_inputs, targets, k):
    """
    For 1 dimensional training data, it produces a function f:reals-> reals
    that outputs the mean training value in the k-Neighbourhood of any input.

    parameters
    ----------
    training_inputs - 1d array (size N) of  data-points from regression dataset
    targets - the associated targets from regression dataset
    k - the number of neighbours on which to base the prediction.

    returns
    -------
    prediction_function - a function that takes 1d array (size M) of test inputs 
      xs and outputs a 1d array of predictions ys, where ys[i] is the prediction
      for test input xs[i]
    """
    # reshape the arrays to support distance calculation
    N = training_inputs.size
    training_inputs = training_inputs.reshape((1,-1))
    def prediction_function(test_inputs):
        test_inputs = test_inputs.reshape((-1,1))
        # uses broadcasting see:
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        distances = np.abs(training_inputs - test_inputs)
        predicts = np.empty(M)
        # each row of each_k_neighbours is the indices of the k
        # neighbours of test_input[i] in training_inputs
        each_k_neighbours = np.argpartition(distances, kth=k, axis=-1)[:,:k]
        for i, neighbourhood in enumerate(each_k_neighbours):
            # the neighbourhood is the indices of the closest training inputs 
            # to test_inputs[i] the prediction is the mean of the targets
            # for this neighbourhood
            predicts[i] = np.mean(targets[neighbourhood])
        return predicts
    # We return a handle to the locally defined function
    return prediction_function

def construct_knn_function(training_inputs, targets, k, metric):
    """
    Produces a function with signature f:R^D-> R
    that outputs the mean training value in the k-Neighbourhood of any D dimensional
    input.
    
    parameters
    ----------
    training_inputs - 2d (N,D)-array of inputs, where N is the number of training
      data-points and D is the dimension of the points (rows) of inputs
    targets - the associated targets from regression dataset
    k - the number of neighbours on whic to base the prediction.
    metric - the distance function which takes 2 2d arrays as input, and 
      produces a matrix of distances between each point (row) in X with each
      point (row) in Y. For instance,

         distances = metric(X, Y) 

      is a valid call if X and Y are both 1d arrays of size (Nx,D) and (Ny,D)
      respectively. This call must produce an 2d output array of distances where
      distances[i,j] equals the distance between X[i,:] and Y[j,:].

    returns
    -------
    prediction_function - a function that takes 2d (M,D)-array of inputs X and 
      outputs a 1d array of predicitons y, where y[i] is the prediction for data
      point X[i,:]
    """
    def prediction_function(test_inputs):
        M, D = test_inputs.shape
        distances = metric(test_inputs, training_inputs)
        predicts = np.empty(M)
        for i, neighbourhood in enumerate(np.argpartition(distances, k)[:,:k]):
            # the neighbourhood is the indices of the closest inputs to xs[i]
            # the prediction is the mean of the targets for this neighbourhood
            predicts[i] = np.mean(targets[neighbourhood])
        return predicts
    # We return a handle to the locally defined function
    return prediction_function


    

