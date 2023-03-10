# -*- coding: utf-8 -*-
"""
Functions that simulates and loads in data for the Variational Autoencoder

Created on Wed Apr 20 14:31:58 2022

@author: MauritsvdOever

"""


def GenerateNormalData(dims, n, correlated_dims, rho):
    """
    Parameters
    ----------
    dims            : int, amount of variables in produced dataset
    n               : int, amount of observations
    correlated_dims : int, amount of factors driving the dataset
    rho             : correlation coefficient between factors

    Returns
    -------
    a n by dims array of correlated normal data (non-diagonal covar matrix)

    """
    import numpy as np
  
    array = np.empty((n,dims))
    
    for variable in range(dims):
        array[:,variable] = np.random.normal(0, 1, n)
        
    amount_of_cols_per_dim = int(dims / correlated_dims)
    
    counter = 0
    
    for i in range(0, correlated_dims):
        for col in range(1, amount_of_cols_per_dim):
            array[:,counter+col] = rho*array[:,counter] + np.sqrt(1-rho**2) * array[:,counter+col]
        counter += amount_of_cols_per_dim
    
    return array


def GenerateStudentTData(dims, n, correlated_dims, rho):
    """
    Parameters
    ----------
    dims            : int, amount of variables in produced dataset
    n               : int, amount of observations
    correlated_dims : int, amount of factors driving the dataset
    rho             : correlation coefficient between factors

    Returns
    -------
    a n by dims array of correlated student t data, with degrees of freedom being random from

    """
    import random
    from copulae import GaussianCopula
    import numpy as np
    from scipy.stats import t
    array = np.zeros((n,dims))
    
    cols_per_dim = int(dims/correlated_dims)
    
    if cols_per_dim == 1:
        for col in range(array.shape[1]):
            array[:,col] = np.random.standard_t(df=random.randint(3,10), size=n)
            
    else:
        counter = 0
        for dim in range(correlated_dims):
            cop = GaussianCopula(dim = cols_per_dim)
            cop.params = np.array([rho]*len(cop.params))
            array[:,counter:counter+cols_per_dim] = cop.random(n)
            counter += cols_per_dim
    
    return array

    
def Yahoo(list_of_ticks, startdate, enddate, retsorclose = 'rets'):
    '''
    Parameters
    ----------
    list_of_ticks : list of strings, tickers
    startdate     : string, format is yyyy-mm-dd
    enddate       : string, format is yyyy-mm-dd
    retsorclose   : string, 'rets' for returns and 'close' for adjusted closing prices
    
    
    Returns
    -------
    dataframe of stock returns or prices based on tickers and date

    '''
    import yfinance as yf
    import pandas as pd
    import numpy as np
    
    dfclose = pd.DataFrame(yf.download(list_of_ticks, start=startdate, end=enddate))['Adj Close']
    dfclose = dfclose.ffill()
    dfclose = dfclose.backfill()
    
    if retsorclose == 'rets':
        dfrets  = np.log(dfclose) - np.log(dfclose.shift(1))
        return dfrets.iloc[1:,:]
    else:
        return dfclose

    

    
    
    
    
    
    
    
    