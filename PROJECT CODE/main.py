#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import seaborn as sns


# In[2]:



# In[3]:


from processing import processing


# In[4]:


from models import LogisticRegression
from models import RandomForest, SVM, KNN
from models import grid_search, LR_lambda_cv, evaluate_model


# In[6]:
def command_parse():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--test_frac",
                            help = "the fraction of dataset used for testing",
                            default = 0.2,
                            type = float)

    argparser.add_argument("--dataset",
                            help = "the csv file to be loaded",
                            default = "winequalityN.csv",
                            type = str)
    argparser.add_argument("--sample_size",
                            help = "sample size to be loaded from the raw dataset",
                            default = 100,
                            type = int)
    
    argparser.add_argument("--model",
                            help = "model to be selected",
                            default = "SVM",
                            type = str)
    
    argparser.add_argument("--cv",
                            help = "number of folds in cv",
                            default = 5,
                            type = int)

    argparser.add_argument("--N",
                            help = "hyperparameter search size",
                            default = 5,
                            type = int)
    


    args = argparser.parse_args()
    return args



def main(args):

    print("Exploring and Processing dataset")
    
    state = 42
    print("[Red wine group]")
    rX_train, ry_train, rX_test, ry_test = processing(args.dataset, args.sample_size, group = "red", test_frac=args.test_frac, state=state)
    
    print("[White wine group]")
    wX_train, wy_train, wX_test, wy_test = processing(args.dataset, args.sample_size, group = "white", test_frac=args.test_frac, state=state)
    
    print("Performing grid search")
    r_val_acc, r_hyper = grid_search(args.model, "red",rX_train, ry_train, cv=args.cv, N=args.N)
    print("The best hyper parameters found for red wine: {}".format(r_hyper))
    w_val_acc, w_hyper = grid_search(args.model,"white", wX_train, wy_train, cv=args.cv, N=args.N)
    print("The best hyper parameters found for white wine: {}".format(w_hyper))

    print("Evaluate the model using best hyperparameters found")
    print("The classification report for red wine:")
    evaluate_model(args.model,"red", rX_train, ry_train, rX_test, ry_test, r_hyper)
    print("The classification report for white wine:")
    evaluate_model(args.model,"white", wX_train, wy_train, wX_test, wy_test, w_hyper)
    #SVM
    
    #Random Forest
    
    #LOGISTIC REGRESSION 
    # print("Running Logistic regression experiments")

    
    
    # LR_lambda_cv(W_validation_inputs,W_validation_targets,W_test_inputs,W_test_targets,wine_type='white wine',lambda_list = np.logspace(-4,-1,10))
    # LR_lambda_cv(R_validation_inputs,R_validation_targets,R_test_inputs,R_test_targets,wine_type='red wine',lambda_list = np.logspace(-4,-1,10))


# In[7]:


if __name__ == '__main__':
    args = command_parse()
    main(args)


# In[ ]:




