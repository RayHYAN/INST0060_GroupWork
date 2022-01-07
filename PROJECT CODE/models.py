#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



from processing import accuracy_score, train_test_split
from evaluation import classificationreport
from evaluation import confusion_matrix
from time import process_time


# In[3]:

class LogisticsRegression():
    def __init__(self, lamda):
        self.lamda = lamda

    def fit(self, X_train, y_train):
        self.weights = logistic_regression_fit(X_train, y_train, lamda=self.lamda)
    
    def predict(self, X_test):
        y_predict = logistic_regression_predict(X_test, self.weights)
        return y_predict

RandomForest = RandomForestClassifier
SVM = lambda C, gamma: SVC(C=C, gamma=gamma, kernel='rbf')

def cross_validation(model, X, y, cv=5):
    if cv == 1:
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_frac=0.2, state=42)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        val_acc = accuracy_score(y_test, y_predict)
        return [val_acc]

    Xfolds = np.array_split(X, cv)
    yfolds = np.array_split(y, cv)
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
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        val_acc = accuracy_score(y_test, y_predict)
        cross_vals.append(val_acc)
    return cross_vals

def grid_search(name, X, y, cv=5, N=5):
    
    svm_hyper = {'C': np.logspace(-4, 1, N), 'gamma': np.logspace(-1, 2, N)}
    rf_hyper = {'n_estimators': [10, 100, 1000], 'max_depth': np.arange(1, 11)}
    logist_hyper = {'C' : np.logspace(-2, 2, N), 'penalty': ['none', 'l1', 'l2', 'elasticnet']}
    m2m = {'SVM': (SVM, svm_hyper), 'RF': (RandomForest, rf_hyper), 'Logistics': (LogisticsRegression, logist_hyper)}
    Model, params = m2m[name]

    akey, bkey = params.keys()
    alphas = params[akey]
    betas = params[bkey]        
    scores = [] # scores with hyperparameter
    
    for alpha in alphas:
        for beta in betas:
            hyper = dict()
            hyper[akey] = alpha
            hyper[bkey] = beta
            model = Model(**hyper)
            avg_score = np.mean(cross_validation(model, X, y, cv=cv))
            print(avg_score,hyper)
            scores.append((avg_score, hyper))
    
    return max(scores, key=lambda ele : ele[0])

def evaluate_model(name, X_train, y_train, X_test, y_test, hyper):
    m2m = {'SVM': (SVM), 'RF': (RandomForest), 'Logistics': (LogisticsRegression)}
    Model = m2m[name]
    model = Model(**hyper)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    # Performance report
    classificationreport(y_test, y_predict)



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
        val_acc = accuracy_score(y_test,y_predict)
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
    print(f"Testing lambda parameter on validation data for {wine_type} consisting of 5 folds")
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
    plt.legend()
    plt.savefig('Lambda_fitting_on_cross_validation_logistic_regression.png', bbox_inches='tight')
    
    print(f'\nBest parameter(lambda) for logistic regression for {wine_type} on validation data is ' + str(lam_max))
    print(f'\nBest average accuracy score for logistic regression for {wine_type} on validation data is ' + str(score_max))
    print('\nNow running logistic regression on test data with best parameters ...')
    
    LR_test_funct(training_validation_inputs,training_validation_targets,test_inputs,test_targets,wine_type=wine_type,lamda = lam_max)
    


# In[6]:


def LR_test_funct(train_val_inputs,train_val_targets,test_inputs,test_targets, wine_type,lamda = 0):
    
    t1_start = process_time()

    weight = logistic_regression_fit(train_val_inputs,train_val_targets,lamda = lamda)
    predicted_wine_targets = logistic_regression_predict(test_inputs,weight)
    
    t1_stop = process_time()
    
    print("Time taken for model to run on test data in seconds: ",t1_stop-t1_start)
    print(f"\n Classification Report for {wine_type} on test data \n\n")
    classificationreport(test_targets, predicted_wine_targets)
    print(f'\n Confusion matrix for {wine_type} on test data \n\n' + str(confusion_matrix(test_targets,predicted_wine_targets)))


# In[ ]:




