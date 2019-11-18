#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:18:40 2019

@author: joeldacosta
"""
import mpmath
from scipy.stats import norm
from scipy.stats import skew 
from scipy.stats import kurtosis
import math
import pickle
from datetime import datetime
import numpy as np,pandas as pd 
from scipy.stats import pearsonr
import csv
#------------------------------------------------------------------------------ 
def save_obj(obj, name ):
    with open('/users/joeldacosta/desktop/obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('/users/joeldacosta/desktop/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def clusterKMeansBase(corr0,maxNumClusters=10,n_init=10): 
    from sklearn.cluster import KMeans 
    from sklearn.metrics import silhouette_samples 
    dist,silh=((1-corr0.fillna(0))/2.)**.5,pd.Series() # distance matrix 
    for init in range(n_init): 
        print("init: " + str(init))
        for i in range(2,maxNumClusters+1): # find optimal num clusters 
            print("i: " + str(i))            
            kmeans_=KMeans(n_clusters=i,n_jobs=1,n_init=1) 
            kmeans_=kmeans_.fit(dist) 
            silh_=silhouette_samples(dist,kmeans_.labels_) 
            stat=(silh_.mean()/silh_.std(),silh.mean()/silh.std()) 
            if np.isnan(stat[1]) or stat[0]>stat[1]: 
                silh,kmeans=silh_,kmeans_ 
                n_clusters = len( np.unique( kmeans.labels_ ) ) 

    newIdx=np.argsort(kmeans.labels_) 
    corr1=corr0.iloc[newIdx] # reorder rows 
    corr1=corr1.iloc[:,newIdx] # reorder columns 
    clstrs={i:corr0.columns[np.where(kmeans.labels_==i)[0] ].tolist() for i in np.unique(kmeans.labels_) } # cluster members 
    silh=pd.Series(silh,index=dist.index) 
    return corr1,clstrs,silh

def makeNewOutputs(corr0,clstrs,clstrs2): 
    from sklearn.metrics import silhouette_samples 
    clstrsNew,newIdx={},[] 
    for i in clstrs.keys(): 
        clstrsNew[len(clstrsNew.keys())]=list(clstrs[i]) 
    for i in clstrs2.keys(): 
        clstrsNew[len(clstrsNew.keys())]=list(clstrs2[i]) 
    map(newIdx.extend, clstrsNew.values()) 
    corrNew=corr0.loc[newIdx,newIdx]
    
    dist=((1-corr0.fillna(0))/2.)**.5 
    kmeans_labels=np.zeros(len(dist.columns)) 
    for i in clstrsNew.keys(): 
        idxs=[dist.index.get_loc(k) for k in clstrsNew[i]] 
        kmeans_labels[idxs]=i 
    silhNew=pd.Series(silhouette_samples(dist,kmeans_labels),index=dist.index) 
    return corrNew,clstrsNew,silhNew 

#------------------------------------------------------------------------------ 
def clusterKMeansTop(corr0,maxNumClusters=10,n_init=10): 
    maxNumClusters = 20
    corr1,clstrs,silh=clusterKMeansBase(corr0,maxNumClusters=min(maxNumClusters,corr0.shape[1]-1),n_init=n_init) 
    #corr1,clstrs,silh=clusterKMeansBase(corr0,maxNumClusters=min(100,corr0.shape[1]-1),n_init=n_init) 
    #corr1,clstrs,silh=clusterKMeansBase(corr0,maxNumClusters=100,n_init=n_init) 


    clusterTstats={i:np.mean(silh[clstrs[i]])/np.std(silh[clstrs[i]]) for i in clstrs.keys()} 
    tStatMean=np.mean(list(clusterTstats.values()))
    redoClusters=[i for i in clusterTstats.keys() if clusterTstats[i]<tStatMean] 
    print("len " + str(len(redoClusters)))
    if len(redoClusters)<=2: 
        return corr1,clstrs,silh 
    else: 
        keysRedo=[];map(keysRedo.extend,[clstrs[i] for i in redoClusters]) 
        corrTmp=corr0.loc[keysRedo,keysRedo] 
        meanRedoTstat=np.mean([clusterTstats[i] for i in redoClusters]) 
        corr2,clstrs2,silh2=clusterKMeansTop(corrTmp, maxNumClusters=min(maxNumClusters,corrTmp.shape[1]-1),n_init=n_init) 
        # Make new outputs, if necessary 
        corrNew,clstrsNew,silhNew=makeNewOutputs(corr0, {i:clstrs[i] for i in clstrs.keys() if i not in redoClusters},clstrs2) 
        newTstatMean=np.mean([np.mean(silhNew[clstrsNew[i]])/np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()]) 
        if newTstatMean<=meanRedoTstat: 
            return corr1,clstrs,silh 
        else: 
            return corrNew,clstrsNew,silhNew

    


def getIVP(cov,use_extended_terms=False): 
    # Compute the minimum-variance portfolio 
    ivp=1./np.diag(cov) 
    if use_extended_terms: 
        n=float(cov.shape[0]) 
        corr=cov2corr(cov) 
        # Obtain average off-diagonal correlation 
        rho=(np.sum(np.sum(corr))-n)/(n**2-n) 
        invSigma=np.sqrt(ivp) 
        ivp-=rho*invSigma*np.sum(invSigma)/(1.+(n-1)*rho) 
    ivp/=ivp.sum() 
    return ivp


        
def Frequency(returns):
    years = len(returns)/365.25
    frequency = len(returns) / years
    return frequency
    
    
def CalculateSR(returns):
    return np.average(returns)/np.std(returns)

def CalculateASR(cluster_indices, all_returns):
    cluster_returns = all_returns[:, cluster_indices]
    cluster_covarr = np.cov(cluster_returns.T)
    cluster_w = getIVP(cluster_covarr, False)
    #print(sum(cluster_w))
    
    Sk = np.matmul(cluster_returns, cluster_w)
    #for j in range(len(Sk)):
        #ktotal = 0
        #for i in range(len(inds)):
            #ktotal += cluster_w[i] * cluster_returns[j,i]
        #print(round(ktotal,10) == round(Sk[j], 10))
    
    #Strategy is such that trades would hyptotheically occur on all dates
    frequency = Frequency(Sk)
    
    est_sr = CalculateSR(Sk)
    
    aSR = est_sr * np.sqrt(frequency)
    
    return aSR
    
def CalculateEV(clusters, all_returns, strategyFrequency):
    all_aSRs = list()

    for k in range(len(clusters)):
        k_aSR = CalculateASR(clusters[k], all_returns)        
        adj_asr = k_aSR if not math.isnan(k_aSR) else 0        
        all_aSRs.append(adj_asr)

    return (np.var(all_aSRs) / strategyFrequency)


def CalculateSRBenchmark(returns, clusters, bestFrequency):
    

    K = len(clusters)
    V = CalculateEV(clusters, returns, bestFrequency)
    
    e = math.e
    y = float(mpmath.euler)

    Z1 = norm.ppf(1-1/K)
    Z2 = norm.ppf(1-1/(K*e))
    SR_star = math.sqrt(V) * ((1-y)*Z1 + y*Z2)
    
    return SR_star

def CalculateDSR(returns, clusters):
    
    strategy_returns = np.sum(returns,axis=0)
    column_index = np.where(strategy_returns == np.max(strategy_returns))[0][0]

    best_strat_returns = returns[:,column_index] 
    best_SR = CalculateSR(best_strat_returns)
    best_Frequency = Frequency(best_strat_returns)
    
    
    SR_star = CalculateSRBenchmark(returns, clusters, best_Frequency)    
    y3 = skew(best_strat_returns)
    y4 = kurtosis(best_strat_returns)
    
    nominator = (best_SR - SR_star)*math.sqrt(len(best_strat_returns - 1))
    denominator = math.sqrt(1 - y3*best_SR + ((y4-1)/4)*math.pow(best_SR,2))

    DSR  = norm.cdf(nominator/denominator)
    return DSR


def writeCoefficientMatric(data):
    
    configs = pd.unique(data['configuration_id'])
    time_start = min(pd.unique(data['time_step']))
    time_end = max(pd.unique(data['time_step']))
    
    df = pd.DataFrame()
    
    for c in configs:
        print(c)
        vals = list(data['total_profit_rate_observed'][data['configuration_id'] == c])
        zeropad = time_end - time_start - len(vals) + 1
        zerovals = list(np.zeros(zeropad))
        df[c] = vals + zerovals
    
    df = df.fillna(0) # with 0s rather than NaNs
    corr_df = df.corr()
    
    corr_df.to_csv(r'/users/joeldacosta/desktop/all_returns_correlation_matrix_rates.csv')
    
    return df


#returns = pd.read_csv('/users/joeldacosta/desktop/all_return_rates.csv') 
#writeCoefficientMatric(returns)

corrMatrix = pd.read_csv(r'/users/joeldacosta/desktop/all_returns_correlation_matrix_rates.csv')
corrMatrix = corrMatrix.fillna(0) # with 0s rather than NaNs
corrMatrix = corrMatrix.loc[:,corrMatrix.columns[1]:corrMatrix.columns[-1]]

#Cluster Run and Save
    corrNew,clstrsNew,silhNew = clusterKMeansTop(corrMatrix)
    save_obj(clstrsNew, "clstrsNew")
    save_obj(silhNew, "silhNew")
    corrNew.to_csv(r'/users/joeldacosta/desktop/obj/corrNew.csv')
    
#Read In Objects
    clstrsNew = load_obj("clstrsNew")
    silhNew = load_obj("silhNew")
    corrNew = pd.read_csv(r'/users/joeldacosta/desktop/obj/corrNew.csv')



dsr = CalculateDSR(returns, clstrsNew)












