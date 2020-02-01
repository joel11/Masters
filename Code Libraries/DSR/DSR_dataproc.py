#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:18:40 2019

@author: joeldacosta
"""

import numpy as np
import pandas as pd 

def save_obj(obj, name ):
    with open('/users/joeldacosta/desktop/obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('/users/joeldacosta/desktop/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def getReturnsDF(data):
    
    configs = pd.unique(data['configuration_id'])
    time_start = min(pd.unique(data['time_step']))
    time_end = max(pd.unique(data['time_step']))
    
    df = pd.DataFrame()
    
    i = 0
    for c in configs:
        print(c)
        vals = list(data['total_profit_rate_observed'][data['configuration_id'] == c])
        zeropad = time_end - time_start - len(vals) + 1
        zerovals = list(np.zeros(zeropad))
        df[i] = vals + zerovals
        i = i + 1
    
    df = df.fillna(0) # with 0s rather than NaNs
    
    return df

def writeCoefficientMatrix(data, filename):
    corr_df = data.corr()
    corr_df.to_csv(filename)
    return corr_df

def getCoefficientMatrix(returns_File, corrFile):
    returns = pd.read_csv(returns_File)
    df_returns = getReturnsDF(returns)
    df_corrMatrix = writeCoefficientMatric(df_returns, corrFile)
    return dr_corrMatrix


