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
import numpy as np

array  = datafuncs.get_data('returns')

#%% cell to define variables

dim_Z  = 3
dist   = 'normal'
epochs = 1000
layers = 1

#%%
# from Models.VAE import VAE

# q = 0.05


# for i in range(20):
#     model = VAE(array, dim_Z, layers=layers, done=False, dist=dist, plot=False)
#     model.fit(epochs)
    
#     model.insample_VaRs(quantile=q, plot=False, output=True)
#     del model


#%% 
from Models.VAE import VAE

q = 0.05

avg_ratio = 0
avg_pval  = 0
for year in np.linspace(2018,2022,5):
    
    ratio = 0
    pval  = 0
    for i in range(20):
        split_date = pd.to_datetime('01-01-'+str(int(year)))
        start_date = split_date - pd.offsets.DateOffset(years=5)
        
        X_train = array.loc[(array.index < split_date) * (array.index > start_date)].copy()
        X_test  = array.loc[array.index > split_date].copy()
        
        model = VAE(X_train, dim_Z, layers=layers)
        model.fit(epochs)
    
        test, (ratio, binom_pval) = model.outofsample_VaRs(X_test, q, output=False, plot=False)
        del model
        if binom_pval > pval:
            pval = binom_pval
            ratio = ratio
    print('ratio of '+str(year)+' = ' + str(ratio))
    avg_ratio += ratio
    avg_pval  += pval
    
print('')
print('')
print('average ratio = ', str(avg_ratio/5))
print('average pval  = ', str(avg_pval/5))
