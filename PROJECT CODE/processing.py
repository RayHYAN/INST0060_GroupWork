#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc


# In[4]:


from fomlads.evaluate.eval_regression import train_and_test_partition
from fomlads.evaluate.eval_regression import train_and_test_filter


# In[5]:


def processing(ifname):
    df = pd.read_csv(r"./winequalityN.csv")

    #Drop any empty values
    df.dropna(inplace = True)

    #We want to partition the data by wine type 
    df_grouped = df.groupby(df.type)
    #Creating a seperate dataframe for red and white wine
    RED_DF = df_grouped.get_group("red")
    WHITE_DF = df_grouped.get_group("white")
    
    return RED_DF, WHITE_DF


# In[6]:


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
            


# In[7]:


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
    


# In[8]:


def accuracy_score(pred_targets,real_targets):
    number_of_targets = len(real_targets)
    score = 0
    
    for i in range(number_of_targets):
        if real_targets[i]==pred_targets[i]:
            score += 1
    accuracy = score/number_of_targets
    return accuracy


# In[9]:



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
    


# In[ ]:


def scalar_funct(inputs):
    """
    This function takes in the input array for both training and testing and standardises all the columns 
    within the array
    
    """
    for i in range(inputs.shape[1]):
        mean_i = np.mean(inputs[:,i])
        std_i = np.std(inputs[:,i])
        inputs[:,i] = (inputs[:,i] - mean_i)/std_i
    return inputs


# In[ ]:



