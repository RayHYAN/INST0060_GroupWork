# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 17:09:07 2022

@author: 龚佳妮
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
import scipy as sc


import scipy.stats
import pandas as pd
from itertools import groupby
from fomlads.data.external import import_for_classification

from fomlads.model.classification import project_data
from fomlads.model.classification import maximum_separation_projection

from fomlads.plot.exploratory import plot_scatter_array_classes
from fomlads.plot.exploratory import plot_class_histograms

FM_WHITE = pd.read_csv("FM_WHITE.csv")
FM_RED = pd.read_csv("FM_RED.csv")



from fomlads.evaluate.eval_regression import train_and_test_filter
from fomlads.evaluate.eval_regression import train_and_test_partition


def dframe_train_test_input_target(dffeature,test_frac):
    """
    This function takes in the new data frame with the feature mapping already applied and converts
    this into training and testing inputs with a split given by the test_fraction. The function should
    output the training inputs and targets as well as the testing inputs and targets. 
    
    
    
    """
    
    np.random.seed(1) #Setting a consistent seed 
    featurematrix = dffeature.to_numpy() #Converting the dataframe into a numpy array 
    
    columns = len(list(dffeature.columns))
    
    inputs = featurematrix[:,:(columns-1)] #We split the matrix into inputs 
    targets = featurematrix[:,columns-1] #Take the last column of the matrix as targets 
    
    train_filter,test_filter=train_and_test_filter(len(dffeature),test_frac) #Applying the training and test split for the inputs and targets using our test fraction
    
    train_inputs, train_targets, test_inputs,test_targets = train_and_test_partition(inputs,targets,train_filter,test_filter) 
    
    return train_inputs,train_targets,test_inputs,test_targets #Returning our training and testing inputs and targets
    

W_train_inputs,W_train_targets,W_test_inputs,W_test_targets = dframe_train_test_input_target(FM_WHITE,0.2) #W stands for white
R_train_inputs,R_train_targets,R_test_inputs,R_test_targets = dframe_train_test_input_target(FM_RED,0.2) #R stands for Red







-------------------------------------------------------------------------------
#Fisher's regression (for red wine)
from fomlads.data.external import import_for_classification

from fomlads.model.classification import project_data
from fomlads.model.classification import maximum_separation_projection

from fomlads.plot.exploratory import plot_scatter_array_classes
from fomlads.plot.exploratory import plot_class_histograms






def project_and_histogram_data(
        inputs, targets, method, title=None, classes=None):
    """
    Takes input and target data for classification and projects
    this down onto 1 dimension according to the given method,
    then histograms the projected data.

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1
    """
    weights = get_projection_weights(inputs, targets, method)
    projected_inputs = project_data(inputs, weights)
    ax = plot_class_histograms(projected_inputs, targets)
    # label x axis
    ax.set_xlabel(r"$\mathbf{w}^T\mathbf{x}$")
    ax.set_title("Projected Data: %s" % method)
    if not classes is None:
        ax.legend(classes)



def get_projection_weights(inputs, targets, method):
    """
    Helper function for project_and_histogram_data
    Takes input and target data for classification and projects
    this down onto 1 dimension according to the given method

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1
    returns
    -------
    weights - the projection vector
    """
    if len(np.unique(targets)) > 2:
        raise ValueError("This method only supports data with two classes")
    if method == 'maximum_separation':
        weights = maximum_separation_projection(inputs, targets)
    elif method == 'fisher':
        weights = fisher_linear_discriminant_projection(inputs, targets)
    else:
        raise ValueError("Unrecognised projection method")
    return weights





from fomlads.model.classification import fisher_linear_discriminant_projection
project_and_histogram_data(inputs=R_train_inputs, targets=R_train_targets, method='fisher')



##eveluate the model
##ROC curve
from fomlads.plot.evaluations import plot_roc
def construct_and_plot_roc(
        inputs, targets, method='maximum_separation', **kwargs):
    """
    Takes input and target data for classification and projects
    this down onto 1 dimension according to the given method,
    then plots roc curve for the data.

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1
    """
    weights = get_projection_weights(inputs, targets, method)
    projected_inputs = project_data(inputs, weights)
    new_ordering = np.argsort(projected_inputs)
    projected_inputs = projected_inputs[new_ordering]
    targets = np.copy(targets[new_ordering])
    N = targets.size
    num_neg = np.sum(1-targets)
    num_pos = np.sum(targets)
    false_positive_rates = np.empty(N)
    true_positive_rates = np.empty(N)
    for i in range(projected_inputs.size):
        false_positive_rates[i] = np.sum(1-targets[i:])/num_neg
        true_positive_rates[i] = np.sum(targets[i:])/num_pos
    fig, ax = plot_roc(
        false_positive_rates, true_positive_rates, label=method, **kwargs)
    return fig, ax



fig,ax = construct_and_plot_roc(
    inputs = R_train_inputs, targets = R_train_targets, method='fisher', colour='b')
ax.legend()

-------------------------------------------------------------------------------
## Model performance evaluation



------------------------------------------------------------------------------------
#Fisher's regression (for white wine)
from fomlads.model.classification import fisher_linear_discriminant_projection
project_and_histogram_data(inputs = W_train_inputs, targets = W_train_targets, method='fisher')

fig,ax = construct_and_plot_roc(
    inputs = W_train_inputs, targets = W_train_targets, method='fisher', colour='b')
ax.legend()

----------------------------------------------------------------------------------
## Model performance evaluation



-----------------------------------------------------------------------------------













## logistic regression model (for red wine)
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy.stats
import pandas as pd
from itertools import groupby


from fomlads.data.external import import_for_classification
from fomlads.plot.exploratory import plot_scatter_array_classes
from fomlads.plot.exploratory import plot_class_histograms
from fomlads.plot.evaluations import plot_roc

from fomlads.model.classification import shared_covariance_model_fit
from fomlads.model.classification import shared_covariance_model_predict
from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict
from fomlads.model.classification import logistic_regression_prediction_probs




def fit_and_plot_roc_logistic_regression(
        inputs, targets, fig_ax=None, colour=None):
    """
    Takes input and target data for classification and fits shared covariance
    model. Then plots the ROC corresponding to the fit model.

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1
    """
    weights = logistic_regression_fit(inputs, targets)
    #
    thresholds = np.linspace(0,1,101)
    N = targets.size
    num_neg = np.sum(1-targets)
    num_pos = np.sum(targets)
    # Comment:
    # There is a good deal of code here that is identical with the function:
    # fit_and_plot_roc_generative, how would you refactor things to ensure the
    # majority is only written once? Consider, a new function with signature:
    # evaluate_and_plot_roc(inputs, prediction_function, thresholds)
    # where prediction_function is a function object that is called with two
    # arguments an input and a threshold. You may need to make use
    # of lambda functions to achieve this though.
    false_positive_rates = np.empty(thresholds.size)
    true_positive_rates = np.empty(thresholds.size)
    for i, threshold in enumerate(thresholds):
        prediction_probs = logistic_regression_prediction_probs(inputs, weights)
        predicts = (prediction_probs > threshold).astype(int)
        num_false_positives = np.sum((predicts == 1) & (targets == 0))
        num_true_positives = np.sum((predicts == 1) & (targets == 1))
        false_positive_rates[i] = np.sum(num_false_positives)/num_neg
        true_positive_rates[i] = np.sum(num_true_positives)/num_pos
    fig, ax = plot_roc(
        false_positive_rates, true_positive_rates, fig_ax=fig_ax, colour=colour)
    # and for the class prior we learnt from the model
    predicts = logistic_regression_predict(inputs, weights)
    fpr = np.sum((predicts == 1) & (targets == 0))/num_neg
    tpr = np.sum((predicts == 1) & (targets == 1))/num_pos
    ax.plot([fpr], [tpr], 'rx', markersize=8, markeredgewidth=2)
    return fig, ax


LR_model = fit_and_plot_roc_logistic_regression(
    inputs = R_train_inputs, targets = R_train_targets, colour='b')



------------------------------------------------------------------------------
# Make predictions




----------------------------------------------------------------------------------
## logistic regression model (white wine)
logistic_model = fit_and_plot_roc_logistic_regression(
    inputs = W_train_inputs, targets = W_train_targets, colour='b')