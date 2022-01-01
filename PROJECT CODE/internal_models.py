# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:36:58 2021

@author: 龚佳妮
"""

# data pre-processing
# we will store our data as a pandas dataset
import pandas as pd

# we will use mat plot for some sexy graphs
import matplotlib.pyplot as plt
%matplotlib inline

# we use numpy for maths and general good stuff 
import numpy as np

# our data set is in an arfff format
from scipy.io import arff
import scipy as sc


import scipy.stats
import pandas as pd
from itertools import groupby

## provided code for Ex. 7.4 a)
from fomlads.data.external import import_for_classification

from fomlads.model.classification import project_data
from fomlads.model.classification import maximum_separation_projection

from fomlads.plot.exploratory import plot_scatter_array_classes
from fomlads.plot.exploratory import plot_class_histograms



#Reading in the data and dropping empty values

import os
os.getcwd()

df = pd.read_csv("winequalityN.csv")
#Drop any empty values
df.dropna(inplace = True)

#We want to partition the data by wine type 
df_grouped = df.groupby(df.type)
#Creating a seperate dataframe for red and white wine
RED_DF = df_grouped.get_group("red")
WHITE_DF = df_grouped.get_group("white")

WHITE_DF


def derived_rep(dframe,size,state):
    """
    Function that pairs a sample against each other through concatenation. It takes a sample size from a
    data frame and then pairs them with all the other entries within the sample size creating a derived 
    representation. Each row in the outputted data frame is the concatanation of two wine samples.
    
    """
    df_sample = dframe.sample(n=size,random_state=state) #We want to extract a random n number of entries from the data frame
    df_sample = df_sample.drop(columns=['type']) #We want to remove the 'type' column since our data frame is already partitioned into white and red
    
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    
    for entry1 in range(len(df_sample)):
        for entry2 in range(entry1):
            df1 = df1.append(df_sample.iloc[entry1].to_dict(),ignore_index=True)
            df2 = df2.append(df_sample.iloc[entry2].to_dict(),ignore_index=True)
            
    new_df = df1.join(df2,lsuffix='_1',rsuffix='_2')
    
    return new_df


white_concat = derived_rep(WHITE_DF,141,1)
red_concat = derived_rep(RED_DF,141,1)




#Feature mapping
def feature_mapping(dframe):
    """
    This function is the feature mapping for the derived representation. It takes the difference between each
    pair of wines and returns the difference between them. Hence we have a feature mapping that takes both 
    raw inputs for items i and j and produces a single feature vector representing the pair of inputs.
    
    We then create the targets using the wine quality, If wine 1's quality is better than wine 2 then we assign
    a 1, if the quality is not greater then we assign it a 0.
    
    """
    fm_dframe = pd.DataFrame()
    fm_dframe['alcohol_diff'] = dframe.apply(lambda x: x['alcohol_1'] - x['alcohol_2'], axis=1)
    fm_dframe['chlorides_diff'] = dframe.apply(lambda x: x['chlorides_1'] - x['chlorides_2'], axis=1)
    fm_dframe['citric acid_diff'] = dframe.apply(lambda x: x['citric acid_1'] - x['citric acid_2'], axis=1)
    fm_dframe['density_diff'] = dframe.apply(lambda x: x['density_1'] - x['density_2'], axis=1)
    fm_dframe['fixed acidity_diff'] = dframe.apply(lambda x: x['fixed acidity_1'] - x['fixed acidity_2'], axis=1)
    fm_dframe['free sulfur dioxide_diff'] = dframe.apply(lambda x: x['free sulfur dioxide_1'] - x['free sulfur dioxide_2'], axis=1)

    fm_dframe['pH_diff'] = dframe.apply(lambda x: x['pH_1'] - x['pH_2'], axis=1)
    fm_dframe['residual sugar_diff'] = dframe.apply(lambda x: x['residual sugar_1'] - x['residual sugar_2'], axis=1)
    fm_dframe['sulphates_diff'] = dframe.apply(lambda x: x['sulphates_1'] - x['sulphates_2'], axis=1)
    fm_dframe['total sulfur dioxide_diff'] = dframe.apply(lambda x: x['total sulfur dioxide_1'] - x['total sulfur dioxide_2'], axis=1)
    fm_dframe['volatile acidity_diff'] = dframe.apply(lambda x: x['volatile acidity_1'] - x['volatile acidity_2'], axis=1)
    fm_dframe['quality_diff'] = dframe.apply(lambda x: x['quality_1'] - x['quality_2'], axis=1)
    
    
    fm_dframe["Target"] = (fm_dframe["quality_diff"] >= 0).astype(int)
    fm_dframe = fm_dframe.drop(columns=['quality_diff'])
    
    return fm_dframe


FM_WHITE = feature_mapping(white_concat)
FM_RED = feature_mapping(red_concat)






from fomlads.evaluate.eval_regression import train_and_test_partition
from fomlads.evaluate.eval_regression import train_and_test_filter


def training_validation_split(data_frame, randomseed=0):
    """
    Inputs the dataframe for either red and white wine and splits it in half with half used for training
    and validation and the other half for testing
    
    """
    training_filter, test_filter = train_and_test_filter(len(data_frame),test_fraction=0.5)
    data_frame_training_validation = data_frame[training_filter]
    data_frame_testing = data_frame[test_filter]
    
    return data_frame_training_validation,data_frame_testing




from fomlads.evaluate.eval_regression import train_and_test_partition
from fomlads.evaluate.eval_regression import train_and_test_filter

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
    




from fomlads.evaluate.eval_regression import train_and_test_partition
from fomlads.evaluate.eval_regression import train_and_test_filter
from fomlads.evaluate.eval_regression import create_cv_folds

def dframe_cross_validation_inputs_target(dffeature,validation_folds):
    """
    This function takes in the training and validation data frame with the feature mapping 
    already applied and converts this into cross validation inputs over 5 folds 
    with a split given by the test_fraction. The function should
    output the training inputs and targets as well as the testing inputs and targets. 
    
    """
    
    np.random.seed(1) #Setting a consistent seed 
    featurematrix = dffeature.to_numpy() #Converting the dataframe into a numpy array 
    
    columns = len(list(dffeature.columns))
    
    inputs = featurematrix[:,:(columns-1)] #We split the matrix into inputs 
    targets = featurematrix[:,columns-1] #Take the last column of the matrix as targets 
    
    folds = create_cv_folds(len(dffeature),validation_folds)
    
    training_inputs = []
    training_targets = []
    validation_inputs = []
    validation_targets = []
    
    train_filter,validation_filter=train_and_test_filter(len(dffeature),test_fraction=0.75) #Applying the training and test split for the inputs and targets using our test fraction
    
    for i in range(validation_folds):
        train_inputs, train_targets, valid_inputs,valid_targets = train_and_test_partition(inputs,targets,train_filter,validation_filter) 
        
        training_inputs.append(train_inputs)
        training_targets.append(train_targets)
        validation_inputs.append(valid_inputs)
        validation_targets.append(valid_targets)
    
    
    return train_inputs,train_targets,validation_inputs,validation_targets #Returning our training and testing inputs and targets


W_inputs,W_targets,W_test_inputs,W_test_targets = dframe_train_test_input_target(FM_WHITE,0.8)#W stands for white
R_inputs,R_targets,R_test_inputs,R_test_targets = dframe_train_test_input_target(FM_RED,0.8) #R stands for Red    







--------------------------------------------------------------------------------------
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
project_and_histogram_data(inputs=R_inputs, targets=R_targets, method='fisher')



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
    inputs = R_inputs, targets = R_targets, method='fisher', colour='b')
ax.legend()

-------------------------------------------------------------------------------
## Model performance evaluation
def accuracy_score(pred_targets,real_targets):
    number_of_targets = len(real_targets)
    score = 0
    
    for i in range(number_of_targets):
        if real_targets[i]==pred_targets[i]:
            score += 1
    accuracy = score/number_of_targets
    return accuracy


R_Weight = fisher_linear_discriminant_projection(R_inputs, R_targets)
projected_wine_targets  = fisher_regression_predict(R_test_inputs,R_Weight)

accuracy = accuracy_score(projected_wine_targets,R_test_targets)

print(accuracy)









-------------------------------------------------------------------------------
#Fisher's regression (for white wine)
from fomlads.model.classification import fisher_linear_discriminant_projection
project_and_histogram_data(inputs = W_train_inputs, targets = W_train_targets, method='fisher')

construct_and_plot_roc(
    inputs = W_train_inputs, targets = W_train_targets, method='fisher', colour='b')
ax.legend()

-------------------------------------------------------------------------------
## Model performance evaluation
from fomlads.model.classification import fisher_linear_discriminant_projection
from fomlads.model.classification import project_data
W_Weight = fisher_linear_discriminant_projection(R_inputs, R_targets)
projected_probs = project_data(R_test_inputs,W_Weight)

def fisher_regression_predict(inputs, weights):
    """
    Get deterministic class prediction vector from the Fisher's regression model.

    parameters
    ----------
    inputs - input data (or design matrix) as 2d array
    weights - a set of model weights
    """
    projected_probs = project_data(inputs, weights)
    return (projected_probs > 0).astype(int)

projected_wine_targets  = fisher_regression_predict(W_test_inputs,W_Weight)

accuracy = accuracy_score(projected_wine_targets,W_test_targets)

print(accuracy)









-------------------------------------------------------------------------------
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



logistic_model = fit_and_plot_roc_logistic_regression(
    inputs = R_inputs, targets = R_targets, colour='b')



------------------------------------------------------------------------------
# Model performance evaluation
from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict
from fomlads.model.classification import logistic_regression_prediction_probs

R_weight = logistic_regression_fit(R_inputs,R_targets)
p = logistic_regression_prediction_probs(W_test_inputs, W_weight)
predicted_wine_targets = logistic_regression_predict(R_test_inputs,R_weight)
accuracy = accuracy_score(predicted_wine_targets,R_test_targets)

print(accuracy)








----------------------------------------------------------------------------------
## logistic regression model (white wine)
logistic_model = fit_and_plot_roc_logistic_regression(
    inputs = W_inputs, targets = W_targets, colour='b')


------------------------------------------------------------------------------
# Make predictions
W_weight = logistic_regression_fit(W_inputs,W_targets)
p = logistic_regression_prediction_probs(W_test_inputs, W_weight)
predicted_wine_targets = logistic_regression_predict(W_test_inputs,W_weight)
accuracy = accuracy_score(predicted_wine_targets,W_test_targets)

print(accuracy)
