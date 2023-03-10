# -*- coding: utf-8 -*-
"""
This repo is created to improve the VAE implementation

Created on Fri Mar 10 15:30:02 2023

@author: MauritsvandenOeverPr
"""
import pandas as pd
import seaborn as sns
from Data import datafuncs 
import matplotlib.pyplot as plt
# from Models import MGARCH
# from Models import VAE


list_of_ticks = ['AMZN', 'BA', 'CMCSA', 'CSCO', 'GOOG', 'JPM', 'MA', 
                 'META', 'NFLX', 'NVDA', 'PEP', 'TSLA', 'TSM', 'V', 'WMT']

startdate       = '2010-01-01'
enddate         = '2023-01-01'
dims            = 12
n               = 10000
correlated_dims = 3
rho             = 0.75

#norm_array  = datafuncs.GenerateNormalData(dims, n, correlated_dims, rho)
t_array     = datafuncs.GenerateStudentTData(dims, n, correlated_dims, rho)
#df_rets    = datafuncs.Yahoo(list_of_ticks, startdate, enddate)


#%% 
df = pd.DataFrame(t_array)
sns.heatmap(df.corr())