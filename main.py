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
import pandas as pd
import seaborn as sns
from Data import datafuncs 
import matplotlib.pyplot as plt
import seaborn as sns

array  = datafuncs.get_data('returns')
dim_Z  = 2
dist   = 'normal'
epochs = 400
layers = 1

#%%
from Models.VAE import VAE

model = VAE(array, dim_Z, layers=layers, done=False, dist=dist, plot=False)
model.fit(epochs)

z = model.encoder(model.X).detach().numpy()

#%%
from Models.VAE import VAE

model = VAE(array, dim_Z, layers=layers, done=False, dist=dist, plot=False)
model.fit(epochs)

test_garchs, test_sigmas = model.fit_garchs()