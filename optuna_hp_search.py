''' Hyperparameter Optimization with Optuna'''
''' Generally, I see that Optuna takes longer to run for the same number of runs. '''

import functools
import pandas as pd 
import numpy as np 

from sklearn import ensemble
from sklearn import metrics 
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline 

import optuna 

def optimize(trial, X, y):
    ''' In Optuna, you need to define a grid of parameters inside of the optimization function directly. '''

    criterion = trial.suggest_categorical('criterion', ['gini','entropy'])
    n_estimators = trial.suggest_int('n_estimators', 100, 1500)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    max_features = trial.suggest_uniform('max_features', 0.01, 1)

    model = ensemble.RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth, 
        max_features = max_features,
        criterion = criterion
    ) 
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

    optimziation_function = functools.partial(optimize, X = X, y = y)

    study = optuna.create_study(direction = 'minimize') #An optuna study corresponds to an optimization task. 
    study.optimize(optimziation_function, n_trials = 15)

    print(study.best_params)