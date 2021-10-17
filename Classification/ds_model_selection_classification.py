# -*- coding: utf-8 -*-
"""

#Getting ready to estimation!

It's used **sklearn version 1.0** in this script. The version of the library defines models and their hyperparameters to estimate. But you can use any version, just put suitable models and parameters or update your sklearn with below command
"""

#!pip install -U scikit-learn

"""#Importing dependencies"""

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

"""#Handle with input data

Choose models and parameters. Put it into model list
"""

params_dt = {'criterion' : ['gini', 'entropy'],
             'splitter': ['best', 'random'],
             'max_depth': [None, 2, 3, 5, 7, 9],
             'max_features':[None, 'auto', 'sqrt', 'log2']}

params_knn = {'n_neighbors': [3, 4, 5, 7],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

params_svc = {'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
              'degree':[2, 3, 4, 5, 6],
              'gamma':['scale', 'auto']}

params_lr = {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
             'fit_intercept': [True, False],
             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

params_gnb = {'var_smoothing':[1e-9, 1e-8, 1e-7]}

params_rc = {'n_estimators': [100, 150, 200, 250],
             'criterion': ['gini', 'entropy'],
             'max_depth': [None, 5, 7, 9],
             'max_features': ['auto', 'sqrt', 'log2']}

# models = [(DecisionTreeClassifier, params_dt), (KNeighborsClassifier, params_knn), 
#           (SVC, params_svc), (LogisticRegression, params_lr), 
#           (GaussianNB, params_gnb), (RandomForestClassifier, params_rc)]
models = [(DecisionTreeClassifier, params_dt), (KNeighborsClassifier, params_knn), 
          (LogisticRegression, params_lr), 
          (GaussianNB, params_gnb), (RandomForestClassifier, params_rc)]

dataset = pd.read_csv('Data_classification.csv')
dataset.head()

# choose the variables from dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# X_train = StandardScaler().fit_transform(X_train.reshape(1, -1))
# y_train = StandardScaler().fit_transform(y_train.reshape(1, -1))

"""#Main function"""

def params_model_selection(model, parameters) -> pd.DataFrame:
    """
    Function to run DS model with different hyperparameters in order to 
    estimate it and choose the one with the highest accuracy
    """
    def combination_params(*params):
        return itertools.product(*params)
    accs = []
    params_final = []
    params_values = list(combination_params(*list(parameters.values())))
    print(f"Estimation parameters of {model.__name__} model")

    for c in tqdm(params_values, position=0, leave=False):
        # print(c)
        try:
            params = dict(zip(tuple(parameters.keys()), c))
            classifier = model(**params)


            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            params_final.append(params)
            # print(f' Params {params_values.index(c)+1}/{len(params_values)} of {model.__name__} model is estimated')
        except ValueError:
            pass
        
    print(f'Estimation {model.__name__} model done!')
    df_res = pd.DataFrame({'Model':model.__name__, 'Accuracy': accs, 'params_final': params_final})
    max_acc = df_res['Accuracy'].max()
    print(f"Max value of accuracy is {max_acc}")
    return df_res

"""#Create result in table"""

df_result = pd.DataFrame(columns=['Model', 'Accuracy', 'params_final'])

for model in models:
    df_result = pd.concat([df_result, params_model_selection(model[0], model[1])])
    print('\n-----------------------')

df_result

df_result[df_result['Accuracy']==df_result['Accuracy'].max()]

list(df_result[df_result['Accuracy']==df_result['Accuracy'].max()]['params_final'])