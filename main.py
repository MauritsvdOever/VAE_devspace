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
# from Models import MGARCH
# from Models import VAE


array = datafuncs.get_data('returns')
