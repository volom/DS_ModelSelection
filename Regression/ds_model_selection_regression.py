# -*- coding: utf-8 -*-

#Getting ready to estimation!
"""
It's used **sklearn version 1.0** in this script. The version of the library defines models and their hyperparameters to estimate. But you can use any version, just put suitable models and parameters or update your sklearn with below command
"""

#!pip install -U scikit-learn

"""#Importing dependencies"""

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

"""#Handle with input data

Choose models and parameters. Put it into model list
"""

params_dt = {'criterion': ('mse', 'friedman_mse', 'absolute_error', 'mae', 'poisson'),
             'max_depth': (1, 2, 3, 4, 5, 6, 7),
             'max_features': [None, 'auto', 'sqrt', 'log2'],
             'min_weight_fraction_leaf': [0, 0.1, 0.2, 0.3]}

params_linear_svr = {'epsilon': [0.1, 0.2, 0.3],
                     'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                     'fit_intercept': [True, False],
                     'max_iter': [1, 2, 3, 4, 5, 6, 7]}
params_rf = {'n_estimators': [90, 100, 150, 200],
             'max_depth': [None, 5, 7, 10, 15],
             'max_features': ['auto', 'sqrt', 'log2'],
             }
params_lr = {'fit_intercept': [True, False],
             'normalize': [True, False],
             'n_jobs': [None, 1, 5, 10, 15],
             'positive': [True, False]}
params_gbr = {'loss': ['squared_error', 'ls'],
              'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
              'n_estimators': [100, 150, 200],
              'criterion':['friedman_mse', 'squared_error', 'mse']}

models = [(DecisionTreeRegressor, params_dt), (LinearSVR, params_linear_svr), 
          (RandomForestRegressor, params_rf), (LinearRegression, params_lr),
          (GradientBoostingRegressor, params_gbr)]

all_r2 = []
dataset = pd.read_csv('Data_regression.csv')
dataset.head()

# choose the variables from dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""#Main function"""

def params_model_selection(model, parameters) -> pd.DataFrame:
    """
    Function to run DS model with different hyperparameters in order to 
    estimate it and choose the one with the highest accuracy
    """
    def combination_params(*params):
        return itertools.product(*params)
    r2_scores = []
    params_final = []
    params_values = list(combination_params(*list(parameters.values())))
    print(f"Estimation parameters of {model.__name__} model")

    for c in tqdm(params_values, position=0, leave=False):
        # print(c)
        params = dict(zip(tuple(parameters.keys()), c))
        regressor = model(**params)
        if model.__name__ == 'SVR':
            X2_train = StandardScaler().fit_transform(X_train)
            y2_train = StandardScaler().fit_transform(y_train)

            regressor.fit(X2_train, y2_train)
            y2_pred = regressor.predict(X2_test)

            r2_scores.append(r2_score(y2_test, y2_pred))
            params_final.append(params)
            # print(f' Params {params_values.index(c)+1}/{len(params_values)} of {model.__name__} model is estimated')
        else:
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)

            r2_scores.append(r2_score(y_test, y_pred))
            params_final.append(params)
            # print(f' Params {params_values.index(c)+1}/{len(params_values)} of {model.__name__} model is estimated')
        
    print(f'Estimation {model.__name__} model done!')
    df_res = pd.DataFrame({'Model':model.__name__, 'r2_scores': r2_scores, 'params_final': params_final})
    max_r2 = df_res['r2_scores'].max()
    print(f"Max value of R2 is {max_r2}")
    return df_res

"""#Create result in table"""

df_result = pd.DataFrame(columns=['Model', 'r2_scores', 'params_final'])

for model in models:
    df_result = pd.concat([df_result, params_model_selection(model[0], model[1])])
    print('\n-----------------------')

df_result

df_result[df_result['r2_scores']==df_result['r2_scores'].max()]

list(df_result[df_result['r2_scores']==df_result['r2_scores'].max()]['params_final'])