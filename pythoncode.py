# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:36:39 2023

@author: daham
"""

#%% importing packages and libraries  
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


#%% data loading and overview for the trainin and the testing sets 
train = pd.read_csv(r"C:\Users\daham\OneDrive\Desktop\binary classification\train.csv\train.csv")
train.head() 

test = pd.read_csv(r"C:\Users\daham\OneDrive\Desktop\binary classification\test.csv\test.csv")
test.head() 


print(train.shape, train.info())
print(test.shape, test.info())


#%% Transforming the desired output labels into binary values 
LabelEncoder = LabelEncoder() 

train['defects'] = LabelEncoder.fit_transform(train['defects'])

print(LabelEncoder.classes_) 


#%% creating the model 

random_Seed = 0 

X = train.drop('defects', axis = 1)
y = train['defects'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_Seed) 


#default model (random forest)

model = RandomForestClassifier(random_state=random_Seed)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(model.get_params())

print(classification_report(y_test, y_pred))

#%% finetunning the model by finding optimal hyperparameters using GridSearchCV

param_grid = {
    'n_estimators' :  [25, 50, 100, 150],
    'max_features' :  ['sqrt', 'log2'], 
    'max_depth' : [3, 6, 9], 
    'max_leaf_nodes': [3, 6, 9]
    }

grid_search = GridSearchCV(RandomForestClassifier(), param_grid)

grid_search.fit(X_train, y_train) 

print(grid_search.best_estimator_)

#%% running the model again with the optimized hyperparameters

#reconfigure the model with the optimized hyperparameters
opt_model = RandomForestClassifier(max_depth=9, max_features='log2', max_leaf_nodes=9, n_estimators=150)
opt_model.fit(X_train, y_train) 

#obtaining the predicted values and model evvaluation
y_predicted = opt_model.predict(X_test)

print(model.get_params())

print(classification_report(y_test, y_pred))


#%% model estimation 

#verify that the test comlumns does not contain the labels column 

test.columns

# predicting the labels for the test set 

test_pred = opt_model.predict(test)

print(test_pred)


# saving the results into a local location 
results = [False if x == 0 else True for x in test_pred]

file_loc = r'C:\Users\daham\OneDrive\Desktop\binary classification\sample_submission.csv'
result_file = pd.read_csv(file_loc).drop(["defects"],axis=1)

results = pd.Series(results, name="defects")
print(results)
final_file = pd.concat([result_file,results],axis = 1)
final_file.to_csv("submission.csv",index=False)






























