#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 13:56:23 2021

@author: btu
"""


import numpy as np
import dill
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import pdb
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import glob

tfk = tf.keras

tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions



#%%




def as_sparse_mat(peakFile,windowsFile):
    chrom_offset=dict(np.loadtxt('chrom_offset.tsv',dtype={'names':('chr','pos'),'formats':('S2',np.int64)}))
    chrom_offset=dict(zip(chrom_offset.keys(),np.cumsum(list(chrom_offset.values()))))
    #peaks=np.load(peakFile)[:-200]
    peaksAll=np.load(peakFile)[:-200]
    sparseListbyTF=[]
    windows=np.load(windowsFile,allow_pickle=True)[:,:2]
    for i in range(len(windows)):
        windows[i][1]+=int(chrom_offset[bytes(windows[i][0].replace('chr',''),'utf-8')])
    peaksAll[peaksAll<0.1]=0
    peaksAll[peaksAll>=0.1]=1
    for n in range(59):
        peaks=peaksAll[:,n][:,np.newaxis]
        
        
        nonZeroIndices=np.argwhere(peaks)
        nonZeroVals=peaks[peaks!=0]

        for i in range(len(nonZeroIndices)):
            try:
                nonZeroIndices[i,0]=int(windows[int(nonZeroIndices[i,0]),1]/200)
            except:
                print(i)
        nonZeroIndices,uniPos=np.unique(nonZeroIndices,axis=0,return_index=1)
        nonZeroVals=nonZeroVals[uniPos].astype(int)

        sparseMat=tf.sparse.SparseTensor(nonZeroIndices[:,[1,0]],nonZeroVals,(peaks.shape[1],int(3500000000/200))) # 200bp windows
        sparseMat=tf.sparse.reorder(sparseMat)
        sparseListbyTF.append(sparseMat)
    return sparseListbyTF

peakList=sorted(glob.glob('full_dataset/*/scFAN_predict_using_K562/*.npy'))
windList=sorted(glob.glob('full_dataset/*/*windows.npy'))
sparseMatList=[]
for peak,win in zip(peakList,windList):
    try:
        sparseMatList.append(as_sparse_mat(peak,win))
        print(peak)
    except:
        print(peak,win)
sparseArr=np.array(sparseMatList)
for i in range(sparseArr.shape[1]):
    
    b=tf.sparse.concat(0,list(sparseArr[:,i]))
    b=tf.cast(b,tf.int8)
    with open (f'k562/K562_TF_{i}','wb') as f:
        dill.dump(b,f)


