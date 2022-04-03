''' Bayesian Hyperparameter Optimization using skopt'''

import pandas as pd 
import numpy as np 

from sklearn import ensemble
from sklearn import metrics 
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline 

import skopt
from skopt import space 

import functools

def optimize(params, param_names, X, y):
    ''' Creating an optimziation function to utilize skopt.gp_minimize. https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html 
    Input: List of parameters to optimize 
    Return: Objective value 
    '''

    params = dict(zip(param_names, params))
    model = ensemble.RandomForestClassifier(**params) 
    kf_cv = model_selection.StratifiedKFold(n_splits = 5) #create a K-fold cross validation strategy. 
    
    accuracies = []
    for idx in kf_cv.split(X = X, y = y): #provides train/test indicies to split data into train/test sets. 
        train_idx, test_idx = idx[0], idx[1]

        #Splitting the data
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        #fit, predict, eval
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fold_acc = metrics.accuracy_score(y_test, preds)
        accuracies.append(fold_acc)

    #We want to MINIMIZE since we're using a skopt.gp_minimize  So we want a high accuracy, but make it negative to minimize. 
    return -1.0 * np.mean(accuracies) # If we were to use something like log-loss, then we wouldn't include the negative. 


if __name__ == '__main__':
    train_df = pd.read_csv('./data/MobilePriceDataset/train.csv')

    X = train_df.drop('price_range', axis = 1).values
    y = train_df.price_range.values 

    # When you're using skopt, you must define a param_space. 
    param_space = [
        space.Integer(3, 15, name = 'max_depth'), 
        space.Integer(100, 600, name = 'n_estimators'),
        space.Real(0.01, 1, prior = 'uniform', name = 'max_features'),
        space.Categorical(['gini','entropy'],  name = 'criterion')
    ]

    param_names = [ 
        'max_depth',
        'n_estimators', 
        'max_features',
        'criterion'
    ]

    optimziation_function = functools.partial(
        optimize, #the function we created. 
        param_names = param_names, #names of the parameters that we're looking to optimize. defined in the param_space. 
        X = X,
        y = y
    )

    results = skopt.gp_minimize(
        func = optimziation_function,
        dimensions = param_space,
        n_calls = 15,       #number of calls to func. 
        n_random_starts = 10,
        verbose = 10
    )

    print(
        dict(zip(param_names, results.x))
    )