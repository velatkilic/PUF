#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions
"""

import numpy as np
import os
import scipy.io as sio

def getData(powname="high", dset="train"):
    '''
    Returns dset ("train" or "test") datasets for a given powname ("high", "medium", "low")
    '''
    cwd = os.getcwd()
    if (dset=="train"):
        # read training data
        dat = np.load(cwd+"//Data//"+powname+"train.npz")
        x   = np.float64(dat["xtrain"]) # predictor expects float
        y   = np.float64(dat["ytrain"]) / (2**14 - 1.)
        y2  = None
    elif (dset=="test"):
        # read test data
        dat = np.load(cwd+"//Data//"+powname+"test.npz")
        x   = np.float64(dat["xtest"]) # predictor expects float
        y   = dat["ytest"]
        y2  = dat["ypuf"]
    else:
        print("Dataset not valid")
        
    # add a new column to x to handle the bias term
    sz = x.shape
    x  = np.concatenate((x,np.ones(shape=(sz[0],1))), axis=1)
    return x, y, y2

def calcFhd(xored, keylen=20000):
    '''
    Calculates fractional Hamming distance between the predictions and the
    ground truth data.
    xored         : ytest^ypred. Assumes 14 bit quantization for ytest and ypred.
    keylen        : Number of bits per key
    out(Nkeys, 14): outputs FHD for each key and bit level
    '''

    # Split data into Nkeys each with keylen size
    xored = xored.ravel()
    sz    = xored.shape
    Nkeys = int(sz[0]/keylen)
    xored = xored[0:Nkeys*keylen]
    xored = xored.reshape((-1, keylen))

    # Calculate FHD for each key
    out  = np.zeros((Nkeys,14))
    for i in range(Nkeys):
        for j in range(14):
            fhd = 0
            for k in range(keylen):
                # add to fhd the jth bit for the data[i,k]
                fhd += np.float64(np.binary_repr(xored[i,k],width=16)[j+2])
            fhd = fhd/keylen # normalize by key length
            out[i,j] = fhd
    return out

def loadPred(powname, model):
    '''
    Load predictions for a given powername and model
    '''
    cwd = os.getcwd()
    return np.load(cwd+"/Predictions/"+powname+"_"+model+".npz")["ypred"]

def loadDataVis(pownames, models):
    cwd = os.getcwd()
    # Load ML predictions and ground truth for all power levels
    data = {} # init dict
    for powname in pownames:
        pow_dict = {}
        
        # load ground truth and save it to a temp dictionary
        _, ytest, ypuf = getData(powname, dset="test")
        temp = {}
        temp["gnd"] = ytest
        temp["fhd"] = calcFhd(ypuf^ytest)
        pow_dict["gnd"] = temp
        
        # add ML predictions to the temp dict
        for model in models:
            print("Calculating FHD for " + powname + " " + model )
            temp         = {}
            ypred        = loadPred(powname, model) # load prediction
            temp["pred"] = ypred
            temp["fhd"]  = calcFhd(ypred^ytest)
            
            pow_dict[model] = temp
             
        data[powname] = pow_dict
    sio.savemat(cwd+"/Predictions/allvisdata.mat", data)