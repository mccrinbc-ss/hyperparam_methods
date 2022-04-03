''' Hyperopt Bayesian Hyperparameter Optimization '''

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

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope 

import functools

def optimize(params, X, y):
    ''' When using Hyperopt, params is going to be a dictionary itself, and we won't need param_names.  '''
    ''' Other than this small change, when you move from skopt to hyperopt, everything in the optimization function stays the same. '''

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
    # with skopt, param_space isn't a dictonary. 
    # Furthermore, with hyperopt, we use hp.quniform 
    param_space = {
        'max_depth'    : scope.int(hp.quniform(label = 'max_depth', low = 3, high = 15, q = 1)), 
        'n_estimators' : scope.int(hp.quniform(label = 'n_estimators', low = 100, high = 600, q = 1)),
        'max_features' : hp.uniform('max_features', 0.01, 1),
        'criterion'    : hp.choice('criterion', ['gini','entropy'])
    }

    optimziation_function = functools.partial(
        optimize, #the function we created. 
        X = X,
        y = y
    )

    trials = Trials()

    results = fmin( #fmin is a hyperopt function 
        fn = optimziation_function,
        space = param_space,
        max_evals = 15,
        trials = trials,
        algo   = tpe.suggest
    )

    print(results)