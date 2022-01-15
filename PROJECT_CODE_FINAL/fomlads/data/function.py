import numpy as np
import numpy.random as random

def simple_sin(inputs):
    """
    A simple sin function with period 1
    """
    return np.sin(2*np.pi*inputs)

def arbitrary_function_1(inputs):
    """
    An arbitrary function to provide an interesting form for regression in 1d
    """
    return inputs*np.sin(2*np.pi*(inputs**2))


def arbitrary_function_2(inputs):
    """
    An arbitrary function to provide an interesting form for regression in 1d
    """
    return np.sin(2*np.pi*(2*inputs-1)**4)


def arbitrary_function_3(inputs):
    """
    An arbitrary function to provide an interesting form for regression in 1d
    """
    return np.cos(np.exp(inputs)) + inputs**2


def saw_function(inputs):
    """
    An arbitrary function to provide an interesting form for regression in 1d
    """
    targets = np.empty(inputs.shape)
    targets[inputs<0.5] = inputs[inputs<0.5]
    targets[inputs>=0.5] = inputs[inputs>=0.5]-1
    return targets

def prob_dens_gaussian(xs, mu, sigma2):
    return 1/np.sqrt(2*np.pi*sigma2)*np.exp(-(xs-mu)**2/(2*sigma2))

def logistic_sigmoid(a):
    """
    Calculates the logistic sigmoid for an individual value or collection of 
    values

    parameters
    ----------
    a - input scalar or array

    """
    return 1/(1+np.exp(-a))


