# -*- coding: utf-8 -*-
"""
This repo is created to improve the VAE implementation

Created on Fri Mar 10 15:30:02 2023

@author: MauritsvandenOeverPr

To do list:
    - fix models
        - implement univariate_garch class into VAE
        - add noise to latent space when optimizing
        - implement out-of-sample methods
            - fit VAE
            - fit GARCHs and store params (and vols?)
            - load out-of-sample data in, for every day filter vol, sim, decode, and take quantile, and store as VaRs
                - for day 1 vol, take the last one of the in-sample data (already stored)
            - then plot VaRs and returns for out-of-sample
            
    - tech demo prep
    - implement grid-search routine
"""
# cell to load packages/get data

import pandas as pd
import seaborn as sns
from Data import datafuncs 
import matplotlib.pyplot as plt
import seaborn as sns

array  = datafuncs.get_data('returns')

#%% cell to define variables

dim_Z  = 3
dist   = 'normal'
epochs = 1000
layers = 2

#%%
from Models.VAE import VAE

q = 0.05


for i in range(20):
    model = VAE(array, dim_Z, layers=layers, done=False, dist=dist, plot=False)
    model.fit(epochs)
    
    model.insample_VaRs(quantile=q, plot=False, output=True)
    del model


#%% 
from Models.VAE import VAE

q = 0.05

split_date = pd.to_datetime('01-01-2022')
start_date = split_date - pd.offsets.DateOffset(years=5)

for i in range(20):
    X_train = array.loc[(array.index < split_date) * (array.index > start_date)].copy()
    X_test  = array.loc[array.index > split_date].copy()
    
    model = VAE(X_train, dim_Z, layers=layers)
    model.fit(epochs)
    
    test = model.outofsample_VaRs(X_test, q, output=True)
