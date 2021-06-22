# -*- coding: utf-8 -*-
"""
Split training/test data and save as numpy files
    - train set has
        xtrain: input patterns
        ytrain: output hadamard measurements (average of 2 measurements)
    
    - test set has
        xtest: input patterns
        ytest: output hadamard measurements (average of 2 measurements)
        ypuf : output hadamard measurements (single puf measurement for comparison)
        
"""
import numpy as np
import h5py
import os

seed = 214
np.random.seed(seed)

per   = 90 # percent training data

dirName = os.getcwd() + '/'

fnames = ['ML_analog_5_28_2019','ML_analog_midPower_6_7_2019','ML_analog_highPower_5_31_2019']
snames = ['low','medium','high']

for i in range(3):
    fname = fnames[i]
    arrays = {}
    f = h5py.File(dirName+fname+'.mat')
    for k, v in f.items():
        arrays[k] = np.array(v)
    
    inpDat = arrays['ML_input']
    outDat = arrays['ML_output']
    
    del arrays
    
    # normalize output -> map [0 1]
    mlmin  = np.min(outDat)
    mlran  = np.max(outDat) - mlmin
    
    outDat = (outDat-mlmin)/mlran;
    
    # average two channels -> ground truth
    gndDat = np.mean(outDat,0)
    
    # Retain only first quarter of data
    inpDat = inpDat[:,0:200000]
    outDat = outDat[:,:,0:200000]
    gndDat = gndDat[:,0:200000]
    
    # Shuffle data indices
    sz    = np.shape(outDat)
    trLen = round(sz[2]*per/100);
    
    ind      = np.random.permutation(sz[2])
    indTrain = ind[0:trLen]
    indTest  = ind[trLen+1:sz[2]]
    
    xtrain = inpDat[:,indTrain]
    ytrain = gndDat[:,indTrain] # train on ground truth
    
    xtest = inpDat[:,indTest]
    ytest = gndDat[:,indTest]
    ypuf  = outDat[0,:,indTest]
    
    
    # Transpose
    xtrain = np.transpose(xtrain)
    ytrain = np.transpose(ytrain)
    
    xtest = np.transpose(xtest)
    ytest = np.transpose(ytest)
    
    
    del inpDat, outDat
    
    # conversion for minimal space
    xtest  = xtest.astype(np.bool_)    # bool input
    xtrain = xtrain.astype(np.bool_)   # bool input
    ytest  = (ytest*(2**14-1)).astype(np.uint16)  # measurements use 14 bit ADC
    ytrain = (ytrain*(2**14-1)).astype(np.uint16) # measurements use 14 bit ADC
    ypuf   = (ypuf*(2**14-1)).astype(np.uint16)   # measurements use 14 bit ADC
    
    np.savez(dirName+snames[i]+'train.npz',xtrain=xtrain,ytrain=ytrain)
    np.savez(dirName+snames[i]+'test.npz',xtest=xtest,ytest=ytest,ypuf=ypuf)