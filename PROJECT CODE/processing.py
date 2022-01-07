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


def processing(file, sample_size, group, test_frac=0.2, state=42, standardized=True):
    df = pd.read_csv(file)

    #Drop any empty values
    df.dropna(inplace = True)

    #We want to partition the data by wine type 
    df_grouped = df.groupby(df.type)

    #Creating a seperate dataframe for red and white wine
    DF = df_grouped.get_group(group)
    print("Deriving the representation...")    
    concat = derived_rep(DF, sample_size, state)
    print("feature mapping...")
    #concat = new_feature_mapping(concat)
    concat = feature_mapping(concat)
    #print(concat.columns)
    X, y = get_dataset(concat)
    print("standardizing...")
    if standardized:
        X = scalar_funct(X)

    print("splitting the dataset...")
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_frac, state=42)

    return X_train, y_train, X_test, y_test


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
        for entry2 in range(len(df_sample)):
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

def new_feature_mapping(frame):
    frame["Target"] = frame.apply(lambda x: 1 if x['quality_1'] - x['quality_2'] > 0 else 0, axis=1)
    frame = frame.drop(columns=['quality_diff']) if 'quality_diff' in frame.columns else frame #Dropping the quality difference column
    frame = frame.drop(columns=['quality_1']) if 'quality_1' in frame.columns else frame
    frame = frame.drop(columns=['quality_2']) if 'quality_2' in frame.columns else frame
    print(frame.columns)
    return frame

def get_dataset(frame):
    

    featurematrix = frame.to_numpy() #Converting the dataframe into a numpy array 
    
    columns = len(list(frame.columns))
    
    X = featurematrix[:,:(columns-1)] #We split the matrix into inputs 
    y = featurematrix[:,columns-1] #Take the last column of the matrix as targets 
    return X, y

# In[8]:


def accuracy_score(predicted_classes,true_classes):
    accuracy = np.sum(np.equal(true_classes, predicted_classes)) / len(true_classes)
    return accuracy


# In[9]:


def train_test_split(X, y, test_frac, state=42):
    np.random.seed(state)
    train_filter,test_filter=train_and_test_filter(len(X),test_frac) #Applying the training and test split for the inputs and targets using our test fraction
    
    X_train, y_train, X_test, y_test = train_and_test_partition(X,y,train_filter,test_filter) 
    
    return X_train, y_train, X_test, y_test #Returning our training and testing inputs and targets


def dframe_train_test_input_target(dffeature,test_frac,state=42):
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
    



