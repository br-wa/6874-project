# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:03:06 2021

@author: clair
"""

#Imports
#import os
import platform
#import random
#import shutil
#import sys
#import functools
#import itertools as it
#import copy

import numpy as np
#import pandas as pd
#import sklearn.metrics
#import h5py

#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
#from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable

#import skimage.measure

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


training_set = None #should have shape [n_TFs, n_cells, n_peaks]
training_labels = None
test_set = None
test_labels = None


#For consistency let's check the python version
print('Python version: {}'.format(platform.sys.version))

# Let's also check the Tensorflow version
tf_version = tf.__version__
print('TensorFlow version: {}'.format(tf_version))

#hyperparams, will tune eventually
n_TFs=10
n_peaks=10045  #173978 #check this for the dataset using
batch_size=12
n_cells=10
latent_dim=4
dropout_rate=.5
l2_lamda=.01
learning_rate=0.001
num_epochs = 1000

###############################################################################
#Data

data_dir='/nobackup/users/btu/shared/dlproj/cstg/'

x_train_6_1000=np.load(data_dir+'x_train_6_cells_1000_pool.npy', allow_pickle=True)
x_val_6_1000=np.load(data_dir+'x_val_6_cells_1000_pool.npy', allow_pickle=True)
x_test_6_1000=np.load(data_dir+'x_test_6_cells_1000_pool.npy', allow_pickle=True)

x_train_6_1000_label=np.load(data_dir+'x_train_6_cells_1000_pool_labels.npy', allow_pickle=True)
x_val_6_1000_label=np.load(data_dir+'x_val_6_cells_1000_pool_labels.npy', allow_pickle=True)
x_test_6_1000_label=np.load(data_dir+'x_test_6_cells_1000_pool_labels.npy', allow_pickle=True)


################################################################################
#Some layers and functions
    
#Linear mesh layer    
class LinearMesh(K.layers.Layer):
    """linear layer, y = x1W1 +x2W2 + b
    
    Properties
    * units: number of neurons (outputs)
      - an `int`
    * activation: output activation function
      - any of ('softmax', 'relu', None)
    * weight_initializer: initialization method for the weights
      - the name of an initializer as a str or a 
        K.initializers.Initializer object
    * bias_initializer: initialization method for the biases
      - the name of an initializer as a str or a 
        K.initializers.Initializer object
    """
    
    def __init__(self, 
                 units, 
                 activation=None,
                 weight_initializer=None,
                 bias_initializer=None):
        """initializes the CustomLinear object"""
        super().__init__()
        self.units = units
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        
        # Declare variables that will be defined later 
        # (you don't have to do this, but it's good style)
        self._w = None
        self._b = None
        self._activation_func = None
    
    def build(self, input_shape):
        """creates the variables of the layer 
        
        this method creates the variables of the layer which will be used
        in the `call` method.
        
        Arguments:
          input_shape: a `TensorShape` object or other iterable which
            specifies the size of the inputs to the `call` method.
        """
        self._w1 = self.add_weight(
            'weight',
            shape=(latent_dim, self.units),
            initializer=self.weight_initializer,
            trainable=True,
            #regularizer=l2,
            dtype='float32')
        
        self._w2 = self.add_weight(
            'weight',
            shape=(latent_dim, self.units),
            initializer=self.weight_initializer,
            #regularizer=l2,
            trainable=True,
            dtype='float32')
        
        self._b = self.add_weight(
            'bias',
            shape=(self.units, ),
            initializer=self.bias_initializer,
            trainable=True,
            dtype='float32')
        
        if self.activation is None:
            self._activation_func = None
            
        elif self.activation == 'relu':
            self._activation_func = tf.nn.relu
            
        elif self.activation == 'softmax':
            self._activation_func = tf.nn.softmax
            
        else:
            err_msg = 'Unexpected activation, "%s", passed.' %  self.activation
            raise ValueError(err_msg)
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """performs the computational logic of the layer
        
        Arguments:
          inputs: the input `tf.Tensor` to this layer
          training: (None) boolean flag representing if the layer is 
            being called in train-time (`True`) or run-time (`False`)
        
        Returns:
          outputs: the computed output `tf.Tensor`
        """
        x1, x2 = inputs
        hidden = tf.matmul(x1, self._w1) + tf.matmul(x2, self._w2) + self._b
        if self._activation_func is not None:
            outputs = self._activation_func(hidden) 
        else:
            outputs = hidden
            
        return outputs
    
    
#Third model
def sparse_reg(activ_matrix):
    p = 0.1
    beta = 1
    p_hat = K.backend.mean(activ_matrix) # average over the batch samples
    print("p_hat = ",p_hat)
    #KLD = p*(K.log(p)-K.log(p_hat)) + (1-p)*(K.log(1-p)-K.log(1-p_hat))
    KLD = p*(K.backend.log(p/p_hat)) + (1-p)*(K.backend.log(1-p/1-p_hat))
    KLD=K.backend.abs(KLD)
    print("KLD = ", KLD)
    return beta * K.backend.sum(KLD) # sum over the layer units

###############################################################################    
#Model
full_inputs = K.Input(shape=(n_cells, n_TFs, n_peaks))


n_TFs=10
n_peaks=10045  #173978 #check this for the dataset using
batch_size=12
n_cells=10
latent_dim=128
dropout_rate=.5
l2_lamda=.01
learning_rate=0.0001
epochs=100


#Encoder A
encoderA_inputs = tf.transpose(full_inputs, perm=[0, 2, 1, 3])
xA = layers.Dense(1024, activation="relu", kernel_regularizer=l2(l2_lamda), activity_regularizer=sparse_reg)(encoderA_inputs)
xA = layers.Reshape((n_TFs, n_cells*1024), input_shape=(n_TFs, n_cells, 1024))(xA)
zA = layers.Dense(latent_dim, activation="relu", kernel_regularizer=l2(l2_lamda), activity_regularizer=sparse_reg)(xA)

#Encoder B
encoderB_inputs = full_inputs
xB = layers.Dense(1024, activation="relu", kernel_regularizer=l2(l2_lamda), activity_regularizer=sparse_reg)(encoderB_inputs)
xB = layers.Reshape((n_cells, n_TFs*1024), input_shape=(n_cells, n_TFs, 1024))(xB)
zB = layers.Dense(latent_dim, activation="relu", kernel_regularizer=l2(l2_lamda), activity_regularizer=sparse_reg)(xB)

#Meshing 
y1 = layers.Dense(latent_dim* n_cells, activation="relu", name='meshing1', activity_regularizer=l2(l2_lamda))(zA)
y1 = tf.keras.layers.Reshape((n_TFs, n_cells, latent_dim), input_shape=(n_TFs, n_cells*latent_dim), name='meshing2')(y1)
y1 = tf.transpose(y1, perm=[0, 2, 1, 3])
y2 = layers.Dense(latent_dim* n_TFs, activation="relu", name='meshing3', activity_regularizer=l2(l2_lamda))(zB)
y2 = tf.keras.layers.Reshape((n_cells, n_TFs, latent_dim), input_shape=(n_cells, n_TFs*latent_dim), name='meshing4')(y2)
y = LinearMesh(latent_dim, activation='relu')([y1, y2])
#y=tf.keras.layers.Add()([y1, y2]) #Figrue out how to weight this and add bias -> maybe make own layer

#Decoder C
xC = layers.Dense(1024, activation="relu", name='decoder2', kernel_regularizer=l2(l2_lamda), activity_regularizer=sparse_reg)(y)
decoderC_outputs = layers.Dense(n_peaks, activation="softmax",  kernel_regularizer=l2(l2_lamda), activity_regularizer=sparse_reg)(xC)

vae3_model = K.Model(inputs=full_inputs, outputs=decoderC_outputs, name="branched_model")
vae3_model.summary()

##############################################################################

def vae_loss_fn(inputs, outputs):
    cce=K.losses.CategoricalCrossentropy()
    #mse = K.losses.MeanSquaredError()
    reconstruction_loss=cce(inputs, outputs)
    #reconstruction_loss *=1
    kl_A_loss = 1 + zA_log_var - K.backend.square(zA_mean) - K.backend.exp(zA_log_var)
    kl_B_loss = 1 + zB_log_var - K.backend.square(zB_mean) - K.backend.exp(zB_log_var)
    kl_A_loss = K.backend.sum(kl_A_loss, axis=-1)*-0.5
    kl_B_loss = K.backend.sum(kl_B_loss, axis=-1)*-0.5
    vae_loss = K.backend.mean(reconstruction_loss + kl_A_loss+kl_B_loss)
    return vae_loss

#vae_model.add_loss(vae_loss)


vae3_model.compile(
    optimizer=tf.keras.optimizers.Adam(.001),
    loss=K.losses.CategoricalCrossentropy())

x_train=x_train_6_1000
x_val=x_val_6_1000
x_test=x_test_6_1000
x_train_bin=x_train
x_train_bin[x_train_bin > 0.1]=1

vae3_model.fit(x_train_bin, x_train_bin,
            epochs=200,
            batch_size=12,
            verbose=1,
            validation_data=(x_val, x_val))

################################################################################
def visualize(data, labels):
    """
    Input: np.array with shape (m,n)
    """
    if len(data[0,:]) > 50:
        pca = PCA(n_components=30)
        data_pca = pca.fit_transform(data) #Using PCA to decrease dimensionality for tSNE
        data_embedded = TSNE(n_components=2, n_iter=2000, perplexity=20, learning_rate=100).fit_transform(data_pca)
    elif len(data[0, :]) == 2:
        data_embedded=data
    else:
        data_pca=data
        data_embedded = TSNE(n_components=2, n_iter=2000, perplexity=20, learning_rate=100).fit_transform(data_pca)
    print(data_embedded.shape)

    #y=np.arange(30)
    #color = [item/30 for item in y]
    
    plt.figure(0)
    plt.scatter(data_embedded[:,0], data_embedded[:,1], c=labels, 
                norm=matplotlib.colors.Normalize(), cmap='gist_rainbow')
    plt.title('t-SNE')
    plt.show()

##############################################################################
x_pred=vae3_model.predict(x_train_bin)
cell_1_pred=x_pred[0:44, :, :, :].reshape((440, -1))
print(cell_1_pred)
visualize(cell_1_pred, x_train_6_1000_label)
print(x_train_bin[0:44, :, :, :].reshape((440, -1)))

cell_1_train=x_train[0:44, :, :, :].reshape((440, -1))
visualize(cell_1_train, x_train_6_1000_label)


###############################################################################
TF_latent_model = K.Model(inputs=full_inputs, outputs=zA, name='tf_latent')
Cell_latent_model = K.Model(inputs=full_inputs, outputs=zB, name='cell_latent')


TF_latent_model.compile()

Cell_latent_model.compile()

tf_latent_test=TF_latent_model.predict(x_train)
tf_latent_1=tf_latent_test[0:12, :, :]
tf_latent_1=tf_latent_1.reshape((120, -1))
np.save(data_dir+'branched_3_tf_latent_1.npy', tf_latent_test)



cell_latent_test=Cell_latent_model.predict(x_train)
arr=np.arange(0, 528, step=12)
cell_latent_split=np.split(cell_latent_test, 528, axis=0)
cell_latent_1=cell_latent_test[44:88, :, :]
split=np.split(cell_latent_1, 44, axis=0)
cell_latent_1=np.squeeze(np.concatenate(split, axis=1))
print(cell_latent_1.shape)
np.save(data_dir+'branched_3_cell_latent_1.npy', cell_latent_test)
cell_latent_1=cell_latent_1.reshape((440, -1))

visualize(tf_latent_1, np.arange(120))
visualize(cell_latent_1, x_train_6_1000_label)
