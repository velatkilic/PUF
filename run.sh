#!/bin/bash
echo "Removing old files"
rm -rf Models Predictions __pycache__ Plots

echo "Running gradient boosted tree models"
python skmodels.py -p medium -k gbtree
python skmodels.py -p low -k gbtree
python skmodels.py -p high -k gbtree

echo "Running sklearn models for low power"
python skmodels.py -p low -k lin_reg
python skmodels.py -p low -k kNN
python skmodels.py -p low -k ran_for

echo "Running sklearn models for medium power"
python skmodels.py -p medium -k lin_reg
python skmodels.py -p medium -k kNN
python skmodels.py -p medium -k ran_for

echo "Running sklearn models for high power"
python skmodels.py -p high -k lin_reg
python skmodels.py -p high -k kNN
python skmodels.py -p high -k ran_for

echo "Running DNN models"
python dnn_train.py -p low	
python dnn_train.py -p medium
python dnn_train.py -p high

echo "Creating plots"
python study_dset_size.py
python viz.py -c 1