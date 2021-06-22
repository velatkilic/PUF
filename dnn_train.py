#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the optimal DNN
"""

import numpy as np
import os
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-bs','--bs', type=int, default=200, help='Batch size')
parser.add_argument('-e','--epoch', type=int, default=200, help='Epoch')
parser.add_argument('-d','--device', type=str, default='cuda:0', help='Device cpu or gpu')
parser.add_argument('-p','--powname', type=str, default='high', help='Pulse power: high, medium, low')
parser.add_argument('-t','--trainModel', type=int, default=1, help='Train dnn or load pretrained')

args = parser.parse_args()

# Parameters
bs         = args.bs
epochs     = args.epoch
device     = args.device
trainModel = args.trainModel
powname    = args.powname

print("Pulse power: " + powname)

cwd    = os.getcwd() # get current directory
params = {"n_layers": 5, "n_units_l0": 138, "n_units_l1": 433, "n_units_l2":306,
          "n_units_l3":501, "n_units_l4":175, "optimizer": "Adam", "lr": 4.991675e-4}
# import data
def getTrainData():
    # train data
    dat = np.load(cwd+"//Data//"+powname+"train.npz")
    x   = np.float32(dat["xtrain"]) # predictor expects double
    y   = np.float32(dat["ytrain"]) / (2**14 - 1.) # normalize to [0 1]
    
    # convert to torch tensor
    x, y = map(torch.tensor, [x, y])
    train_dl = DataLoader(TensorDataset(x, y), batch_size=bs)
    return train_dl

# model
def getModel(best_params):
    n_layers = best_params["n_layers"]
    layers = []

    in_features = 336
    for i in range(n_layers):
        out_features = best_params["n_units_l{}".format(i)]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    layers.append(nn.Linear(in_features, 127))

    return nn.Sequential(*layers)

# objective function which measures network performance
def train(best_params):
    model = getModel(best_params)
    # dataparallel network for a multi gpu setup
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # optimizers
    opt_name = best_params["optimizer"]
    lr  = best_params["lr"]
    opt = getattr(torch.optim, opt_name)(model.parameters(), lr=lr)
    
    # get training dataloader
    train_loader = getTrainData()
    
    # Training of the model.
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.view(data.size(0), -1).to(device), target.to(device)

            opt.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            opt.step()
        
        print(loss)

    return model
if __name__=="__main__":
    
    if args.trainModel==1:
        model = train(params)
        try:
            os.mkdir(cwd + "/Models/")
        except:
            print("Models folder already exists")
        
        torch.save(model.state_dict(), cwd + "/Models/" + powname + "_dnn.pt")
    else:
        model = getModel(params)
        model.load_state_dict(torch.load(cwd + "/Models/" + powname + "_dnn.pt"))
        model.to(device)
    
    # import test data
    dat   = np.load(cwd+"//Data//"+powname+"test.npz")
    xtest = torch.tensor(np.float32(dat["xtest"]), device=device)
    ytest = dat["ytest"]
    del dat
    
    ypred = model(xtest)
    ypred = ypred.to("cpu").data.numpy()
    
    # convert predictions to 14 bit unsigned integer
    ypred = np.uint16(ypred*(2**14-1))
    np.savez(cwd+"/Predictions/"+powname+"_dnn.npz", ypred=ypred)
    
    print("% error "+ str(mean_absolute_percentage_error(ytest,ypred)) + "\n\n")