''' Hyperparameter Optimization with a Basic Grid or Random Search Method '''

import pandas as pd 
import numpy as np 

from sklearn import ensemble
from sklearn import metrics 
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline 

if __name__ == '__main__':
    train_df = pd.read_csv('./data/MobilePriceDataset/train.csv')
    test_df  = pd.read_csv('./data/MobilePriceDataset/test.csv')

    X_train = train_df.drop('price_range', axis = 1).values
    y_train = train_df.price_range.values 
    # X_test  = test_df.drop(['id'], axis = 1).values
    # y_test  = test_df.price_range.values 

 # Preprocessing steps. 
 # These preprocessing steps result in a model that's pretty trash. Just using them to show pipeline functionality. 
    sc = preprocessing.StandardScaler() 
    pca = decomposition.PCA() 

    # classifier = ensemble.RandomForestClassifier(n_jobs = -1) #use a random forest with all CPU cores. Don't necessarily need to scale the data. 
    rf = ensemble.RandomForestClassifier(n_jobs = -1)

 # Optionally, we can transform the random forest classifier above into a full pipeline. 
    classifier = pipeline.Pipeline(
        [
            ('StandardScaling', sc),
            ('pca', pca), #running pca on the data with an unspecified number of components, and we search for the appropriate amount below. 
            ('rf', rf)
        ]
    )

    grid_search = False 
    random_search = True 

    if grid_search: 
        param_grid = {
            "n_estimators" : [100, 200, 300, 400],
            "max_depth"    : [1,2,3,7], 
            "criterion"    : ['gini','entropy']
        }
        model = model_selection.GridSearchCV( 
            estimator = classifier, 
            param_grid = param_grid, 
            scoring = "accuracy", #In this dataset, there is no class imbalance. Accuracy is a good metric here. 
            verbose = 10, 
            n_jobs = 1,
            cv = 5 #5-fold CV, but not stratified. 
        )

    elif random_search: 
        #you must  specify parameters in the pipeline that are specific to a preprocessing step by using a {name}__ key. Name is what you defined in the pipeline 
        param_grid = {
            "pca__n_components": np.arange(5,10), 
            "rf__n_estimators" : np.arange(100, 1500, 100),
            "rf__max_depth"    : np.arange(1,20), 
            "rf__criterion"    : ['gini','entropy']
        }
        model = model_selection.RandomizedSearchCV( 
            estimator = classifier, 
            param_distributions = param_grid, #you can also utilize a distribution, rather than a grid specified over small lists. 
            scoring = "accuracy", #In this dataset, there is no class imbalance. Accuracy is a good metric here. 
            verbose = 10, 
            n_jobs = 1,
            cv = 5, #5-fold CV, but not stratified. 
            n_iter = 10 #run random search n number of times. 
        )

    model.fit(X_train, y_train)

    print(model.best_score_)
    print(model.best_estimator_.get_params())