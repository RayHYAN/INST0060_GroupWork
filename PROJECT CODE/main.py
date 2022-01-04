#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc


# In[2]:


from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict


# In[3]:


from processing import processing
from processing import derived_rep
from processing import feature_mapping
from processing import accuracy_score
from processing import dframe_train_test_input_target
from processing import scalar_funct


# In[4]:


from models import cross_validation_LR
from models import evaluate_cv_LR
from models import LR_lambda_cv
from models import LR_test_funct


# In[5]:


from evaluation import classificationreport
from evaluation import confusion_matrix


# In[6]:


def main():
    print("Processing dataset")
    RED , WHITE = processing(ifname = "red")
    print("Creating new derived representation")
    WHITE_DF = derived_rep(WHITE,141,1)
    RED_DF = derived_rep(RED,141,1)
    
    print("Applying feature mapping")
    FM_WHITE = feature_mapping(WHITE_DF)
    FM_RED = feature_mapping(RED_DF)
    
    print("Creating training/validation and test inputs and targets")
    W_validation_inputs,W_validation_targets,W_test_inputs,W_test_targets = dframe_train_test_input_target(FM_WHITE,0.2)#W stands for white

    R_validation_inputs,R_validation_targets,R_test_inputs,R_test_targets = dframe_train_test_input_target(FM_RED,0.2) #R stands for Red
    #SVM
    
    #Random Forest
    
    #LOGISTIC REGRESSION 
    print("Running Logistic regression experiments")
    scalar_funct(W_validation_inputs)
    scalar_funct(W_test_inputs)
    scalar_funct(R_validation_inputs)
    scalar_funct(R_test_inputs)

    LR_lambda_cv(W_validation_inputs,W_validation_targets,W_test_inputs,W_test_targets,wine_type='white wine',lambda_list = np.logspace(-3,-1,10))
    LR_lambda_cv(R_validation_inputs,R_validation_targets,R_test_inputs,R_test_targets,wine_type='red wine',lambda_list = np.logspace(-3,-1,10))


# In[7]:


main()


# In[ ]:




