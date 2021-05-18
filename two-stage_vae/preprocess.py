#!/usr/bin/env python3
import numpy as np
import dill
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import pdb
import tensorflow_probability as tfp
import glob
import sys
import os
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers

#%%
def as_sparse_mat(peakFile,windowsFile,col=1):
    chrom_offset=dict(np.loadtxt('/nobackup/btu/shared/dlproj/chrom_offset.tsv',dtype={'names':('chr','pos'),'formats':('S2',np.int64)}))
    chrom_offset=dict(zip(chrom_offset.keys(),np.cumsum(list(chrom_offset.values()))))

    peaks=np.load(peakFile)[:-200]

    peaks[peaks<0.1]=0
    peaks[peaks>=0.1]=1
    windows=np.load(windowsFile,allow_pickle=True)[:,:2]
    for i in range(len(windows)):
        windows[i][1]+=int(chrom_offset[bytes(windows[i][0].replace('chr',''),'utf-8')])
    nonZeroIndices=np.argwhere(peaks)
    nonZeroVals=peaks[peaks!=0]

    for i in range(len(nonZeroIndices)):
        try:
            nonZeroIndices[i,0]=int(windows[int(nonZeroIndices[i,0]),1]/1)
        except:
            print(i)
    nonZeroIndices,uniPos=np.unique(nonZeroIndices,axis=0,return_index=1)
    nonZeroVals=nonZeroVals[uniPos].astype(int)
 
    sparseMat=tf.sparse.SparseTensor(nonZeroIndices[:,[1,0]],nonZeroVals,(peaks.shape[1],int(3500000000/1)))
    sparseMat=tf.sparse.reorder(sparseMat)
    return sparseMat
#%%
inFileList=glob.glob(f'{sys.argv[1]}/*')

for i,infile in enumerate(inFileList):
    if os.path.exists(f'{infile}_cprsd') or 'cprsd' in infile:
        continue	
    with open(infile,'rb') as f:
        b=dill.load(f)
    b=tf.cast(b,tf.int8)
    
    
    with tf.device('cpu'):
        intermediate_tensor = tf.sparse.reduce_sum(b, 0)
        zero_vector = tf.ones(b.shape[1], dtype=tf.int8)
        bool_mask = tf.math.greater(intermediate_tensor, zero_vector)
        b=tf.sparse.to_dense(b)
        b=tf.boolean_mask(b,bool_mask,1)
        zero = tf.constant(0, dtype=tf.int8)
        indices = tf.where(tf.not_equal(b, zero))
        values = tf.gather_nd(b, indices)
        sparse = tf.SparseTensor(indices, values, b.shape)
    
    with open (f'{infile}_cprsd','wb') as f:
        dill.dump(sparse,f)
    print(i)
