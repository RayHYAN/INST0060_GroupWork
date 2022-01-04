#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict
from processing import accuracy_score
from evaluation import classificationreport
from evaluation import confusion_matrix
from time import process_time


# In[3]:


# Function for cross validation on logistic regression
def cross_validation_LR(X, y, cv,lamda=0):
    """
    X = training_validation_inputs
    y = training_validation_targets
    
    """
    np.random.seed(1)
    Xfolds = np.split(X, cv)
    yfolds = np.split(y, cv)
    cross_vals = []
    for fold in range(cv):
        y_test = yfolds[fold]
        X_test = Xfolds[fold]
        
        X_train_temp = Xfolds[:fold]
        X_train_temp.extend(Xfolds[fold+1:])
        X_train = np.vstack(X_train_temp)
        y_train_temp = yfolds[:fold]
        y_train_temp.extend(yfolds[fold+1:])
        y_train = np.hstack(y_train_temp)
        weights = logistic_regression_fit(X_train, y_train,lamda=lamda)
        y_predict = logistic_regression_predict(X_test,weights)
        val_acc = accuracy_score(y_test, y_predict)
        cross_vals.append(val_acc)
    return cross_vals


# In[4]:


def evaluate_cv_LR(X, y,lamda=0, output=False):
        
    scores = cross_validation_LR(X, y, cv=5,lamda=lamda)
    avg_cv = np.mean(scores)
    var_cv = np.std(scores)
    if output:
        print(model)
        print("avg cv score is:", avg_cv, "std is:", var_cv)
    return avg_cv, var_cv


# In[5]:


def LR_lambda_cv(training_validation_inputs,training_validation_targets,test_inputs,test_targets,wine_type,lambda_list):
    """
    This function finds the best parameter for lambda on logistic regression then runs the found parameters
    on the test data. It takes in a list of lambdas and tests the values provided within the range. 
    It plots a graph for both red and white wine showing the best value of lambda for both partitions. It then runs the model with the
    best parameters on the test data and outputs the precision, recall  f1 score , and accuracy as well as timing the process.
    
    """
    accuracy_list = []
    print("Testing lambda parameter on validation data consisting of 5 folds")
    for lam in lambda_list:

        val_acc,var_cv=evaluate_cv_LR(training_validation_inputs,training_validation_targets,lamda=lam)
        accuracy_list.append(val_acc)
    
    score_max = max(accuracy_list)
    lam_max = lambda_list[accuracy_list.index(score_max)]
    
    plt.plot(lambda_list, accuracy_list, label = f'{wine_type}')
    
    plt.plot(lam_max, score_max, 'o', color = 'orange')
    
    plt.xlabel('Lambda')
    plt.xscale("log")
    plt.ylabel('Accuracy')
    plt.title("Average Logistic regression accuracies on values of lambda on validation data")
    plt.savefig('foo.png', bbox_inches='tight')
    plt.legend()
    print(f'Best parameter(lambda) for logistic regression for {wine_type} on validation data is ' + str(lam_max))
    print(f'Best average accuracy score for logistic regression for {wine_type} on validation data is ' + str(score_max))
    print('\nNow running logistic regression on test data with best parameters ...')
    
    LR_test_funct(training_validation_inputs,training_validation_targets,test_inputs,test_targets,wine_type=wine_type,lamda = lam_max)
    


# In[6]:


def LR_test_funct(train_val_inputs,train_val_targets,test_inputs,test_targets, wine_type,lamda = 0):
    
    t1_start = process_time()

    weight = logistic_regression_fit(train_val_inputs,train_val_targets,lamda = lamda)
    predicted_wine_targets = logistic_regression_predict(test_inputs,weight)
    accuracy = accuracy_score(predicted_wine_targets,test_targets)
    
    t1_stop = process_time()
    
    print("Time taken for model to run on test data in seconds: ",t1_stop-t1_start)
    print(f"\n Classification Report for {wine_type} on test data \n\n")
    classificationreport(test_targets, predicted_wine_targets)
    print(f'\n Confusion matrix for {wine_type} on test data \n\n' + str(confusion_matrix(test_targets,predicted_wine_targets)))


# In[ ]:




