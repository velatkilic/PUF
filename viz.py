#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize data for paper
"""
import numpy as np
import scipy.io as sio
import os

from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import loadDataVis

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c','--calcData', type=int, default=False,
                    help='Calculate FHD data for visualization if 1')
args = parser.parse_args()

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = "large"
# rc('text', usetex=True)

###############################################################################
# Load Data
cwd = os.getcwd()

# 3 different laser power levels
pownames = ["low", "medium", "high"]

# ML model keys
# models   = ["lin_reg", "ridge", "ela_net", "lasso", "kNN", "ran_for", "gbtree", "dnn"]
models   = ["lin_reg", "kNN", "ran_for", "gbtree", "dnn"]

# Load data dictionary for all powers and models
if args.calcData==1:
    print("Calculating FHD from prediction data. This might take a while.")
    loadDataVis(pownames, models)

data    = sio.loadmat(cwd+"/Predictions/allvisdata.mat")
fhd_vec = np.load(cwd+"/Predictions/train_size_study.npz")["fhd_vec"]
tsizes  = np.load(cwd+"/Predictions/train_size_study.npz")["tsizes"]

def getFHD(powname, model):
    return data[powname][model][0,0]["fhd"][0,0]

def getGnd(powname):
    return data[powname]["gnd"][0,0]["fhd"][0,0]

def privInfo(mu1, va1, mu2, va2):
    # division by np.log(2) to convert from nats to bits
    out = ((va1 - va2)**2 + (va1 + va2)* (mu1-mu2)**2)/(4* np.log(2) * va1 * va2 )
    return out
###############################################################################
# Visualizations

try:
    os.mkdir(cwd+"/Plots/")
except:
    print("Plots folder already exists")

xbits = np.linspace(0,13,14)     # bit level 0 to 14 for average FHD plots
bins  = np.linspace(0,0.5,100)   # histogram bins
xs    = (bins[:-1] + bins[1:])/2 # box centers from edge values for bar plots

# Ground truth data
high_gnd = getGnd("high")
med_gnd  = getGnd("medium")
low_gnd  = getGnd("low")

print("Plotting average FHD and high power pulse FHD histograms")

# Model vs Ground truth
models   = ["kNN","ran_for", "lin_reg",  "gbtree", "dnn"]
labels   = ["kNN", "FOR", "LIN", "GBT", "DNN"]

nModels  = len(models)
# fig = plt.figure(figsize=(10,5*nModels))
# gs  = gridspec.GridSpec(nModels, 2, width_ratios=[1.5,2]) # gridspec with 1-to-2 subplot size ratio

for i in range(nModels):
    model = models[i]
    label = labels[i]
    
    high_mod = getFHD("high", model)
    med_mod  = getFHD("medium", model)
    low_mod  = getFHD("low", model)
    
    # plot average FHD on ax0 and high power FHD histogram on ax1
    fig = plt.figure(figsize=(10,5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1,1.3]) # gridspec with 1-to-2 subplot size ratio
    
    # ax0 = fig.add_subplot(gs[i,0]) # axis for the 0th subplot (for average FHD)
    ax0 = fig.add_subplot(gs[0]) # axis for the 0th subplot (for average FHD)
    ax0.set_xlabel("Bit level (MSB to LSB)")
    ax0.set_ylabel("Average FHD")
    ax0.set_xticks(xbits)
    # predictions (average FHD)
    ax0.plot(xbits, np.mean(high_mod,0), linewidth=2, color='b', marker='v', label='5.0 pJ '+label)
    ax0.plot(xbits, np.mean(med_mod,0), linewidth=2, color='g', marker='v', label='1.5 pJ '+label)
    ax0.plot(xbits, np.mean(low_mod,0), linewidth=2, color='r', marker='v', label='0.5 pJ '+label)
    # ground truth (average FHD)
    ax0.plot(xbits, np.mean(high_gnd,0), linewidth=2, color='b', marker='o', label='5.0 pJ PUF')
    ax0.plot(xbits, np.mean(med_gnd,0), linewidth=2, color='g', marker='o', label='1.5 pJ PUF')
    ax0.plot(xbits, np.mean(low_gnd,0), linewidth=2, color='r', marker='o', label='0.5 pJ PUF')
    ax0.legend()
    ax0.annotate(text="a)",xy = (-0.2, 1.05),xycoords='axes fraction', fontsize=14, fontweight='bold', verticalalignment='top')
    ax0.set_ylim(0,0.55)
    
    # ax1 = fig.add_subplot(gs[i,1], projection='3d')
    ax1 = fig.add_subplot(gs[1], projection='3d')
    # ax.view_init(60, 35)
    ax1.set_xlabel('FHD')
    ax1.set_ylabel('Bit level (MSB to LSB)')
    ax1.set_zlabel('Number of samples')
    for i in range(1,8):
        hist_m, _ = np.histogram(high_mod[:,i], bins = bins)
        hist_g, _ = np.histogram(high_gnd[:,i], bins = bins)
        # add a legend/label only for the last plot since rest will be the same
        if i==7:
            ax1.bar(xs, hist_m, width=0.005, zs=i, zdir='y', color='darkblue', ec='darkblue', alpha=0.8, label='5.0 pJ '+label)
            ax1.bar(xs, hist_g, width=0.005, zs=i, zdir='y', color='dodgerblue', ec='dodgerblue', alpha=0.8, label='5.0 pJ PUF')
        else:
            ax1.bar(xs, hist_m, width=0.005, zs=i, zdir='y', color='darkblue', ec='darkblue', alpha=0.8)
            ax1.bar(xs, hist_g, width=0.005, zs=i, zdir='y', color='dodgerblue', ec='dodgerblue', alpha=0.8)
    ax1.annotate(text="b)",xy = (-0.1, 1.05),xycoords='axes fraction', fontsize=14, fontweight='bold', verticalalignment='top')
    ax1.legend()
    plt.savefig(cwd+"/Plots/"+model+"_fhd.png", dpi=300)
    plt.savefig(cwd+"/Plots/"+model+"_fhd.svg")
# plt.savefig(cwd+"/Plots/model_fhd.png", dpi=300)
# plt.savefig(cwd+"/Plots/model_fhd.svg")
######################################
# Plot all to show the best
titles = ["0.5 pJ Pulse", "1.5 pJ Pulse", "5.0 pJ Pulse"]
fig = plt.figure(figsize=(10,5))
for idx, powname in enumerate(pownames):
    ax = fig.add_subplot(1,3,idx+1)
    for idy, model in enumerate(models):
        fhd = np.mean(data[powname][model][0,0]["fhd"][0,0],0)
        ax.plot(xbits, fhd, linewidth=2, marker='v', label = labels[idy])
    if idx==2: ax.legend()
    if idx==1: ax.set_xlabel("Bit level (MSB to LSB)")
    if idx==0: ax.set_ylabel("Average FHD")
    ax.set_xlim(1,8)
    ax.set_xticks(np.linspace(1,8,8))
    ax.set_ylim(0,0.55)
    ax.set_title(titles[idx])
plt.savefig(cwd+"/Plots/all_models.png", dpi=300)
plt.savefig(cwd+"/Plots/all_models.svg")
######################################
# Plot CRP size data
titles = ["0.5 pJ Pulse", "1.5 pJ Pulse", "5.0 pJ Pulse"]
bits   = np.linspace(1,4,4, dtype=np.uint32)

fig = plt.figure(figsize=(10,5))
for idx, powname in enumerate(pownames):
    ax = fig.add_subplot(1,3,idx+1)
    for bit in bits:
        ax.semilogx(tsizes, fhd_vec[idx,:,bit], label = "Bit "  +str(bit))
    if idx==2: ax.legend()
    if idx==1: ax.set_xlabel("Training Set Size")
    if idx==0: ax.set_ylabel("Average FHD")
    ax.set_title(titles[idx])
    ax.set_ylim(0,0.55)
plt.savefig(cwd+"/Plots/dset_size.png", dpi=300)
plt.savefig(cwd+"/Plots/dset_size.svg")
######################################
# Plot private information
labels   = ["5.0 pJ Pulse", "1.5 pJ Pulse", "0.5 pJ Pulse"]
pownames = ["high", "medium", "low"] 
colors   = ["b", "g", "r"]
fig = plt.figure(figsize=(7,5))
for powname,label,color in zip(pownames, labels, colors):
    fhd = getFHD(powname, "dnn")
    mu1 = np.mean(fhd, 0)
    va1 = (np.std(fhd, 0))**2
    
    gnd = getGnd(powname)
    mu2 = np.mean(gnd, 0)
    va2 = (np.std(gnd, 0))**2
    
    piv = privInfo(mu1, va1, mu2, va2)
    
    plt.plot(xbits, piv, label=label, marker="*", color=color, linewidth=2)
plt.legend()
plt.xlim(0,9)
plt.xlabel("Bit Level (MSB to LSB)")
plt.ylabel("Private Information (bits)")
plt.savefig(cwd+"/Plots/priv_info.png", dpi=300)
plt.savefig(cwd+"/Plots/priv_info.svg")
######################################
# # Compare linear models
# "lin_reg", "ridge", "ela_net", "lasso"
# xticks = np.linspace(1,8,8)

# fig = plt.figure(figsize=(10,5))
# ax1 = fig.add_subplot(131)
# ax1.plot(xbits, np.mean(getFHD("high", "lin_reg"), 0), label="LIN")
# ax1.plot(xbits, np.mean(getFHD("high", "ridge"), 0), label="RID")
# ax1.plot(xbits, np.mean(getFHD("high", "lasso"), 0), label="LAS")
# ax1.plot(xbits, np.mean(getFHD("high", "ela_net"), 0), label="ELA")
# ax1.set_xlabel("Bit level (MSB to LSB)")
# ax1.set_ylabel("Fractional Hamming Distance (FHD)")
# ax1.set_xticks(xticks)

# ax2 = fig.add_subplot(132)
# ax2.set_xlabel("Bit level (MSB to LSB)")
# ax2.set_xticks(xticks)

# ax3 = fig.add_subplot(133)
# ax3.set_xlabel("Bit level (MSB to LSB)")
# ax3.set_xticks(xticks)

