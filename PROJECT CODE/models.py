#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import seaborn as sns

from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from processing import accuracy_score, train_test_split
from evaluation import classificationreport
from evaluation import confusion_matrix
from evaluation import plot_cm
from time import process_time


# In[3]:

class LogisticRegression():
    #Defining a new Class for Logistic Regressi
    
    def __init__(self, lamda, lr):
        self.lamda = lamda
        #self.add_bias_term = add_bias_term
        self.lr = lr

    def fit(self, X_train, y_train):
        self.weights = logistic_regression_fit(X_train, y_train, lamda=self.lamda,lr=self.lr)
    
    def predict(self, X_test):
        y_predict = logistic_regression_predict(X_test, self.weights)
        return y_predict



RandomForest = RandomForestClassifier 
SVM = lambda C, gamma: SVC(C=C, gamma=gamma, kernel='rbf')
KNN = KNeighborsClassifier

def cross_validation(model, X, y, cv=4):
    """
    This function implements the cross validation for a model across folds. The number of folds is specified by the value of cv.
    It takes an input of a model, its training+validation data and the number of folds and outputs the score of the model over each fold.
    
    """
    if cv == 1: #If only 1 fold is required then just train on the data itself with no folds.
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_frac=0.2, state=42)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        val_acc = accuracy_score(y_test, y_predict)
        return [val_acc]

    Xfolds = np.array_split(X, cv)#Split the inputs into number of folds
    yfolds = np.array_split(y, cv)
    cross_vals = []
    for fold in range(cv): #We need repeat the training and validation on each fold
        y_test = yfolds[fold]
        X_test = Xfolds[fold]
        
        X_train_temp = Xfolds[:fold] #Ensuring we train on N-1 folds and validate on remaining fold.
        X_train_temp.extend(Xfolds[fold+1:])
        X_train = np.vstack(X_train_temp)
        
        y_train_temp = yfolds[:fold]
        y_train_temp.extend(yfolds[fold+1:])
        y_train = np.hstack(y_train_temp)
        
        model.fit(X_train, y_train)#Training the model on the fold selected

        y_predict = model.predict(X_test)#Validating on the remaining fold
        val_acc = accuracy_score(y_test, y_predict)#Finding the accuracy
        cross_vals.append(val_acc)#Appending the accuracy to a list
        
    return cross_vals

def grid_search(name,group, X, y, cv=4, N=5):
    """
    This function allows us to test two parameters for each model using cross validation. It takes an input of the wine type, the model,
    the number of folds, and the range of parameters to test. It also saves a heatmap of the scores of each parameter averaged across the folds
    as well as the best parameter found by the grid search.
    

    """
    #Defining the parameters for each model    
    svm_hyper = {'C': np.logspace(-1, 1, N), 'gamma': np.logspace(-2, 1, N)}
    rf_hyper = {'n_estimators': np.arange(10, 50), 'max_depth': np.arange(1, 11)}
    logist_hyper = {'lamda' : np.logspace(-4, -3, 3), 'lr': np.arange(1,5)*0.1}
    knn_hyper = {'n_neighbors' : np.arange(1, 50), 'weights': ['uniform', 'distance']}

    #Defining a dictionary for each model and its parameters
    m2m = {'SVM': (SVM, svm_hyper), 'RF': (RandomForest, rf_hyper),  'KNN': (KNeighborsClassifier, knn_hyper),'Logistic': (LogisticRegression, logist_hyper)}

    Model, params = m2m[name]

    akey, bkey = params.keys()
    alphas = params[akey] #First parameter to test
    betas = params[bkey]  #Second parameter to test       
    scores = [] # scores with hyperparameter
    twoDscores = np.zeros((len(alphas), len(betas)))
    
    #We need to test every combination of both parameters to ensure the optimal hyperparameters have been selected.
    for i in range(len(alphas)):
        for j in range(len(betas)):
            alpha = alphas[i]
            beta = betas[j]
            hyper = dict()
            hyper[akey] = alpha
            hyper[bkey] = beta
            model = Model(**hyper)
            avg_score = np.mean(cross_validation(model, X, y, cv=cv))
            #print(avg_score,hyper)
            scores.append((avg_score, hyper))
            twoDscores[i,j] = avg_score
    
    plt.clf()

    sns.heatmap(twoDscores, annot=True, xticklabels=betas, yticklabels=alphas) #Saving a heat map with the scores for each parameter
    plt.savefig('Hyperparameter tuning heatmap for ' + name + ' for ' + group + ' wine.png' )

    return max(scores, key=lambda ele : ele[0]) #Returning the optimal hyperparamers

def evaluate_model(name , group, X_train, y_train, X_test, y_test, hyper):
    """
    This function retrains the model on the training+validation data with the optimal hyper parameters found
    and then evaluates this on the test data, returning the final classification report for it. 

    """
    m2m = {'SVM': (SVM), 'RF': (RandomForest), 'KNN': (KNeighborsClassifier), 'Logistic': (LogisticRegression)} #Defining the models
    Model = m2m[name]
    model = Model(**hyper) #Using the optimal hyperparameters found 
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    # Performance report
    plt.clf()
    plot_cm(y_test, y_predict,name,group)
    classificationreport(y_test, y_predict)






