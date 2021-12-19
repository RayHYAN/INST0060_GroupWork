# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

# a package for machine learning
import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = pd.read_csv("C:/Users/Ziqi Xu/Desktop/Machine Learning Project(JNG)/Wine Data.csv")
data.dropna(inplace = True)
data["new Type"] = (data["type"] == "white").astype(int)
data.reset_index(inplace = True, drop = True)
# Study the Distribution of column 'quality'. This is highly imbalance.
counting_numbers = []
for i in range(3, 10, 1):
    count_temp = 0
    for index in range(0, data.__len__(), 1):
        if data["quality"][index] == i:
            count_temp += 1
    counting_numbers.append(count_temp)    
            
# Train a binary classification model.
  # Change labels
data["new Label"] = (data["quality"] > 5.5).astype(int)

# Generate the training & test dataset with the help of sklearn.model_selection.train_test_split()
data_columns = list(data.columns)
X = data[["new Type"] + data_columns[1 : 11]]
y = data["new Label"]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 1)

# Train the model
# Generate an empty model
LR_model = LogisticRegression()
# Train the model to get beta parameters
LR_model.fit(X_train, y_train)
# Make predictions
y_pred_proba = LR_model.predict_proba(X_test)
y_pred_label = LR_model.predict(X_test)
# Model performance evaluation
print("The model accuracy score is ", accuracy_score(y_test, y_pred_label), "!")












