#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sklearn models:
    - Linear regression
    - Ridge regression
    - kNN
    - Random forest
    - xgboost
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from utils import getData

from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

from tune_sklearn import TuneSearchCV

import pickle


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-k','--key', type=str, default='lin_reg', help='Model to run')
parser.add_argument('-p','--powname', type=str, default='high', help='Pulse power: high, medium, low')

args = parser.parse_args()

cwd = os.getcwd() # get current directory

powname = args.powname # 3 different laser power levels
key     = args.key

if __name__=="__main__":
    # Model objects dict
    models   = {"lin_reg":linear_model.LinearRegression(n_jobs=-1),
                "ridge":TuneSearchCV(linear_model.Ridge(),
                                      {'alpha':(1e-8,1, 'log-uniform')},
                                      search_optimization="hyperopt"),
                "ela_net":TuneSearchCV(linear_model.MultiTaskElasticNet(),
                                      {'alpha':(1e-8,1, 'log-uniform'),
                                       'l1_ratio':(0,1,'uniform')},
                                      search_optimization="hyperopt"),
                "lasso":TuneSearchCV(linear_model.MultiTaskLasso(),
                                      {'alpha':(1e-8,1e2, 'log-uniform')},
                                      search_optimization="hyperopt"),
                "kNN":KNeighborsRegressor(n_neighbors=100,algorithm="brute"),
                "ran_for":RandomForestRegressor(),
                "gbtree": MultiOutputRegressor(XGBRegressor(booster="gbtree"))
                }

    # Create a folder for the results and the models
    try:
        os.mkdir(cwd+"/Predictions/")
    except:
        print("Predictions folder already exists")

    try:
        os.mkdir(cwd+"/Models/")
    except:
        print("Models folder already exists")

    # iterate thorugh different power levels and ML models
    t1 = time.time()

    print("Loading data for power: " + powname)
    xtrain, ytrain, _ = getData(powname, dset="train")
    xtest, ytest, _   = getData(powname, dset="test")

    print("Data loaded, fitting models for: " + powname)

    print("Fitting: " + key )
    clf = models[key]
    clf.fit(xtrain,ytrain)
    try:
        print("Optimal parameters: "+str(clf.best_estimator))
    except:
        print("No hyperparameter tuning for " + key)

    # predict
    ypred = clf.predict(xtest)

    # convert predictions to 16 bit unsigned integer
    ypred = np.uint16(ypred*(2**14-1))

    print("% error for "+ key + ": " +
          str(mean_absolute_percentage_error(ytest,ypred)) + "\n\n")

    np.savez(cwd+"/Predictions/"+powname+"_"+key+".npz", ypred=ypred)
    pickle.dump(clf, open(cwd+"/Models/"+powname+"_"+key+".sk", 'wb'))

    t2 = time.time()

    print("Elapsed time: " + str((t2-t1)/60.))
