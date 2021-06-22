#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Study how much performance degrades with reduced dataset size
"""
import numpy as np
import os
from utils import getData, calcFhd

from sklearn import linear_model

pownames = ["low", "medium", "high"]
cwd      = os.getcwd()
Ntrials  = 20 # number of trials
tsizes   = np.logspace(2.2, 5.25, Ntrials, dtype=np.int32) # train dataset size
fhd_vec  = np.zeros((3,Ntrials,14)) # init for 3 power levels, Ntrials and 14 bits

for i, powname in enumerate(pownames):
    print(powname)
    xtrain, ytrain, _ = getData(powname, dset="train")
    xtest, ytest, _   = getData(powname, dset="test")
    sz = xtrain.shape
    for j in range(Ntrials):
        print("Training " + str(j))
        # Train
        indz = np.random.randint(0, sz[0], tsizes[j])
        clf  = linear_model.LinearRegression(n_jobs=-1)
        clf.fit(xtrain[indz,:],ytrain[indz,:])
        
        print("Prediction and FHD calculation")
        # Predict
        ypred = clf.predict(xtest)
        ypred = np.uint16(ypred*(2**14-1))
        fhd   = calcFhd(ypred^ytest)
        fhd_vec[i,j,:] = np.mean(fhd,0)
        

np.savez(cwd+"/Predictions/train_size_study.npz", fhd_vec=fhd_vec, tsizes=tsizes)

    
