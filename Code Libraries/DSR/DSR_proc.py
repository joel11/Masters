#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:18:40 2019

@author: joeldacosta
"""

import os
os.chdir('/Users/joeldacosta/Desktop/DSR')
import DSR_dataproc as dataproc
import DSR_PradoLewis as dsr
import numpy as np,pandas as pd 
import csv

def Frequency(returns):
    years = len(returns)/365.25
    frequency = len(returns) / years
    return frequency
    
    
def CalculateSR(returns):
    return np.average(returns)/np.std(returns)

def CalculateASR(cluster_indices, df_returns):
        
    zerolist = list()
    new_indices = list()

    for i in range(len(cluster_indices)):
        tot = sum(df_returns.iloc[:,cluster_indices[i]])
        if tot == 0:
            zerolist.append(cluster_indices[i])
    
    for i in range(len(cluster_indices)):
        if (not zerolist.__contains__(cluster_indices[i])):
            new_indices.append(cluster_indices[i])
    
#   len(new_indices)
#   len(zerolist)
#   len(cluster_indices)
    
    
#    cluster_returns = df_returns[cluster_indices]
    cluster_returns = df_returns[new_indices]
    cluster_covarr = np.cov(cluster_returns.T)


    cluster_w = getIVP(cluster_covarr, False)
    
    Sk = np.matmul(cluster_returns, cluster_w)
    #for j in range(len(Sk)):
        #ktotal = 0
        #for i in range(len(inds)):
            #ktotal += cluster_w[i] * cluster_returns[j,i]
        #print(round(ktotal,10) == round(Sk[j], 10))
    
    #Strategy is such that trades would hyptotheically occur on all dates
    frequency = Frequency(Sk)
    
    est_sr = CalculateSR(Sk)
    print(est_sr)
    
    aSR = est_sr * np.sqrt(frequency)
    
    return aSR
    
def CalculateEV(clusters, df_returns, bestFrequency):
    
    all_aSRs = list()

    for k in range(len(clusters)):
        k_aSR = CalculateASR(clusters[k], df_returns)        
        adj_asr = k_aSR if not math.isnan(k_aSR) else 0        
        all_aSRs.append(adj_asr)

    return (np.var(all_aSRs) / bestFrequency)


def CalculateSRBenchmark(df_returns, clusters, bestFrequency):
    
    clusters = clstrsNew
    bestFrequency = best_Frequency

    K = len(clusters)
    V = CalculateEV(clusters, df_returns, bestFrequency)
    
    e = math.e
    y = float(mpmath.euler)

    Z1 = norm.ppf(1-1/K)
    Z2 = norm.ppf(1-1/(K*e))
    SR_star = math.sqrt(V) * ((1-y)*Z1 + y*Z2)

    return SR_star

def CalculateDSR(returns, clstrsNew, column_index):
    
    #column_index = 1770 - 1    
    #tr = returns[returns['configuration_id'] == 30649]['total_profit_rate_observed']
    #tr = tr.fillna(0)
    #CalculateSR(tr)
    
    
    ###
        
    
    best_strat_returns = df_returns[column_index] 
    best_SR = CalculateSR(best_strat_returns)
    best_SR
    best_Frequency = Frequency(best_strat_returns)
    
    SR_star = CalculateSRBenchmark(df_returns, clstrsNew, best_Frequency)    
    print(SR_star)
    print(best_SR)
    y3 = skew(best_strat_returns)
    y4 = kurtosis(best_strat_returns)
    
    nominator = (best_SR - SR_star)*math.sqrt(len(best_strat_returns) - 1)
    denominator = math.sqrt(1 - y3*best_SR + ((y4-1)/4)*math.pow(best_SR,2))

    DSR  = norm.cdf(nominator/denominator)
    DSR
    return DSR


#returns_File = str(r'/users/joeldacosta/desktop/all_return_rates_cost.csv')
#corrFile = str(r'/users/joeldacosta/desktop/all_return_rates_cost_correlation_matrix_rates.csv')
returns_File = str(r'/users/joeldacosta/desktop/actual_full_return_rates_cost.csv')
corrFile = str(r'/users/joeldacosta/desktop/actual_full_return_rates_cost_correlation_matrix_rates.csv')

returns = pd.read_csv(returns_File) 
df_returns = dataproc.getReturnsDF(returns)
df_corrMatrix = dataproc.writeCoefficientMatrix(df_returns, corrFile)

#Cluster Run and Save
corrNew,clstrsNew,silhNew = dsr.clusterKMeansTop(df_corrMatrix)

len(clstrsNew[0])
len(clstrsNew[1])

(14267 + 7765) - (14400 + 7253)

#column_index = 18182
#dsr = CalculateDSR(subset_df_rates, clstrsNew, column_index)
 
configs = pd.unique(returns['configuration_id'])
#len(clstrsNew[0])
config_string = "select deltas, count(*) from dataset_config where configuration_id in ("
for c in clstrsNew[1]:
    config_string = config_string + str(configs[c]) + ", "

config_string = config_string[0:-2] + ") group by deltas"

print(config_string)


