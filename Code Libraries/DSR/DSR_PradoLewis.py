#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:18:40 2019

@author: Marcos Lopez de Prado, Michael J Lewis

The original paper, and source of this code, can be found at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017

Some superficial changes, such as process visibility and hardcoded values, have been made. It is otherwise as presented in paper.

"""

import numpy as np,pandas as pd 

#------------------------------------------------------------------------------ 

def clusterKMeansBase(corr0,maxNumClusters=10,n_init=10): 
    from sklearn.cluster import KMeans 
    from sklearn.metrics import silhouette_samples 

    maxNumClusters= 10
    n_init = 10
    
    dist,silh=((1-corr0.fillna(0))/2.)**.5,pd.Series() # distance matrix 
    for init in range(n_init): 
        
        print("init: " + str(init))
        
        for i in range(2,maxNumClusters+1): # find optimal num clusters 
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