#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep neural network with optuna for hyperparameter tuning

Parts of the code from optuna pytorch examples:
    https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import optuna
from optuna.trial import TrialState

import argparse

# Parameters
parser = argparse.ArgumentParser()

parser.add_argument('-bs','--bs', type=int, default=200, help='Batch size')
parser.add_argument('-e','--epoch', type=int, default=200, help='Epoch')
parser.add_argument('-d','--device', type=str, default='cuda:0', help='Device cpu or gpu')
parser.add_argument('-p','--powname', type=str, default='high', help='Pulse power: high, medium, low')

args = parser.parse_args()
bs         = args.bs
epochs     = args.epoch
device     = args.device
powname    = args.powname
cwd        = os.getcwd() # get current directory

print("Searching DNN params for " + powname)

# import data
def getTrainData():
    dat = np.load(cwd+"//Data//"+powname+"train.npz")
    x   = np.float32(dat["xtrain"]) # predictor expects double
    y   = np.float32(dat["ytrain"]) / (2**14 - 1.) # normalize to [0 1]
    # Split train and validation data
    xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.1, random_state=1234)
    # convert to torch tensor
    xtrain, xvalid, ytrain, yvalid = map(torch.tensor, [xtrain, xvalid, ytrain, yvalid])
    train_dl = DataLoader(TensorDataset(xtrain, ytrain), batch_size=bs)
    valid_dl = DataLoader(TensorDataset(xvalid, yvalid), batch_size=bs)
    return train_dl, valid_dl

# network model (modified from optuna pytorch example)
def getModel(trial):
    n_layers = trial.suggest_int("n_layers", 3, 6)
    layers = []

    in_features = 336
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 128, 512)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())

        in_features = out_features
    layers.append(nn.Linear(in_features, 127))

    return nn.Sequential(*layers)

# objective function which measures network performance
def objective(trial):
    model = getModel(trial)
    # dataparallel network for a multi gpu setup
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # optimizers
    opt_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr  = trial.suggest_float("lr", 1e-4, 1e1, log=True)
    opt = getattr(torch.optim, opt_name)(model.parameters(), lr=lr)

    # get dataloaders
    train_loader, valid_loader = getTrainData()

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

        # Validation of the model.
        model.eval()
        val_loss = 0
        cnt = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                data, target = data.view(data.size(0), -1).to(device), target.to(device)
                output = model(data)
                loss = F.mse_loss(output, target)
                val_loss += loss
                cnt += 1

        avg_loss = val_loss/cnt

        trial.report(avg_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=100))
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
