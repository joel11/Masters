#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:18:40 2019

@author: joeldacosta
"""

import os
import math
import mpmath
from scipy.stats import norm
from scipy.stats import skew 
from scipy.stats import kurtosis
os.chdir('/Users/joeldacosta/Masters/Code Libraries/DSR')
import DSR_PradoLewis as dsr_pl
import DSR_dataproc as dsr_dataproc
import numpy as np,pandas as pd 

def Frequency(returns):
    years = len(returns)/365.25
    frequency = len(returns) / years
    return frequency
    
    
def CalculateSR(returns):
    return np.average(returns)/np.std(returns)

def CalculateASR(cluster_indices, returns):
        
    #Certain networks with exploding or vanishing gradients result in NULL sharpe ratios causing computation issues. These are removed
    
    zerolist = list()
    new_indices = list()

    for i in range(len(cluster_indices)):
        tot = sum(returns.iloc[:,cluster_indices[i]])
        if tot == 0:
            zerolist.append(cluster_indices[i])
    
    for i in range(len(cluster_indices)):
        if (not zerolist.__contains__(cluster_indices[i])):
            new_indices.append(cluster_indices[i])
        
    cluster_returns = returns[new_indices]
    cluster_covarr = np.cov(cluster_returns.T)
    cluster_w = dsr_pl.getIVP(cluster_covarr, False)
    
    Sk = np.matmul(cluster_returns, cluster_w)
    #for j in range(len(Sk)):
        #ktotal = 0
        #for i in range(len(inds)):
            #ktotal += cluster_w[i] * cluster_returns[j,i]
        #print(round(ktotal,10) == round(Sk[j], 10))
    
    #Strategy is such that trades would hyptotheically occur on all dates
    frequency = Frequency(Sk)
    
    est_sr = CalculateSR(Sk)
    print("Cluster SR: " + str(est_sr))
    
    aSR = est_sr * np.sqrt(frequency)
    
    return aSR
    
def CalculateEV(clusters, returns, bestFrequency):
    
    all_aSRs = list()

    for k in range(len(clusters)):
        k_aSR = CalculateASR(clusters[k], returns)        
        adj_asr = k_aSR if not math.isnan(k_aSR) else 0        
        all_aSRs.append(adj_asr)

    print("Annualized SRs: " + str(all_aSRs))

    var_Clusters = (np.var(all_aSRs) / bestFrequency)
    
    print("Variance of Clustered Trials: " + str(var_Clusters))

    return var_Clusters

def CalculateSRBenchmark(clusters, returns, bestFrequency):
    
    K = len(clusters)
    V = CalculateEV(clusters, returns, bestFrequency)
    
    e = math.e
    y = float(mpmath.euler)

    Z1 = norm.ppf(1-1/K)
    Z2 = norm.ppf(1-1/(K*e))
    SR_star = math.sqrt(V) * ((1-y)*Z1 + y*Z2)

    print("Benchmark SR*: " + str(SR_star))

    return SR_star

def CalculateDSR(returns, clstrsNew):
    
    ## Determine the best strategy and its SR
    maxSR = 0
    maxI = -1

    for i in range(returns.shape[1]):
        curSR = CalculateSR(returns[[i]])[i]
        if(curSR >= maxSR):
            maxSR = curSR
            maxI = i    

    print("Highest SR: " + str(maxSR))
    
    best_strat_returns = returns[maxI] 
    best_Frequency = Frequency(best_strat_returns)
    
    SR_star = CalculateSRBenchmark(clstrsNew, returns, best_Frequency)    
    y3 = skew(best_strat_returns)
    y4 = kurtosis(best_strat_returns)
    
    nominator = (maxSR - SR_star)*math.sqrt(len(best_strat_returns) - 1)
    denominator = math.sqrt(1 - y3*maxSR + ((y4-1)/4)*math.pow(maxSR,2))

    DSR  = norm.cdf(nominator/denominator)

    print("y3: " + str(y3))
    print("y4: " + str(y4))
    print("Nominator: " + str(nominator))
    print("Denominator: " + str(denominator))
    print("DSR: " + str(DSR))

    return DSR


#returns_File = str(r'/users/joeldacosta/desktop/all_return_rates_cost.csv')
#corrFile = str(r'/users/joeldacosta/desktop/all_return_rates_cost_correlation_matrix_rates.csv')
#corrFile = str(r'/users/joeldacosta/desktop/actual_full_return_rates_cost_correlation_matrix_rates.csv')




##Cluster Run and Save
returns_File = str(r'/users/joeldacosta/desktop/actual_full_return_rates_cost.csv')
returns_data = pd.read_csv(returns_File) 
df_returns = dsr_dataproc.getReturnsDF(returns_data)
df_corrMatrix = dsr_dataproc.getCoefficientMatrix(df_returns)#, corrFile)
corrNew,clstrsNew,silhNew = dsr_pl.clusterKMeansTop(df_corrMatrix)
dsr_value = CalculateDSR(df_returns, clstrsNew)



sql_scripts = dsr_dataproc.generateSQLInsert(returns, clstrsNew)


len(clstrsNew[0])
len(clstrsNew[1])













