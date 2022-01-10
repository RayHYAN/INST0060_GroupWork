#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import seaborn as sns


# In[2]:


# Confusion Matrix Function 

def confusion_matrix(true_classes,predicted_classes):

    classes = set(true_classes)
    confusionmatrix  = pd.DataFrame(
        np.zeros((2,2),dtype=int),
        index=classes,
        columns=classes)

    for true_label, prediction in zip(true_classes ,predicted_classes):
        confusionmatrix.loc[true_label, prediction] += 1

    return confusionmatrix.values 

# Plot Confusion Matrix Function 
def plot_cm(true_classes,predicted_classes,group):
    cm_matrix_svc_w = pd.DataFrame(data=confusion_matrix(true_classes,predicted_classes), columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
    
    sns.heatmap(cm_matrix_svc_w, annot=True, fmt='d',cmap = "Blues")
    
    plt.savefig("Confusion Matrix for " + group + " wine")

# In[3]:


# Function for classification Report 
def classificationreport(true_classes,predicted_classes):
    cm = confusion_matrix(true_classes,predicted_classes)
    recall = cm[0,0]/(cm[0,0]+cm[0,1])
    precision = cm[0,0]/(cm[0,0]+cm[1,0])
    f1_score = 2*precision*recall / (precision+recall)
    acc = np.sum(np.equal(true_classes, predicted_classes)) / len(true_classes)
   
    print('Precison is: {0:0.4f}'. format(precision))
    print('Recall is: {0:0.4f}'. format(recall))
    print('F1_Score is: {0:0.4f}'. format(f1_score))
    print('Accuracy is: {0:0.4f}'. format(acc))


# In[ ]:




