# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:02:43 2021

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
learning_rate=0.00005
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
###############################################################################    
#Model

#For a branched model want to use the keras functional api (instead of sequential)

prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(latent_dim), scale=1), 
                                      reinterpreted_batch_ndims=1)

full_inputs = K.Input(shape=(n_cells, n_TFs, n_peaks))



#Layers

#A branch encoder for TF latent space
encoderA_inputs = tf.transpose(full_inputs, perm=[0, 2, 1, 3])
xA = layers.Lambda(lambda x: tf.cast(x, tf.float32))(encoderA_inputs)
xA = layers.Dense(1048, activation="relu", activity_regularizer=l2(l2_lamda))(xA)
xA = layers.Dense(256, activation='relu', activity_regularizer=l2(l2_lamda))(xA)
xA = layers.Dropout(dropout_rate)(xA)
xA = layers.Dense(64, activation="relu", activity_regularizer=l2(l2_lamda))(xA)
xA = layers.Reshape((n_TFs, n_cells*64), input_shape=(n_TFs, n_cells, 64), activity_regularizer=l2(l2_lamda))(xA)
#xA = layers.Dense(16, activation="relu")(xA)
zA_mean = layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim),activation=None)(xA)
zA =  tfp.layers.MultivariateNormalTriL(latent_dim, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior))(zA_mean)
#encoderA = keras.Model(encoder_inputs, zA, name="encoder")
#encoderA.summary()


#B branch encoder for cell latent space
encoderB_inputs=full_inputs
xB = layers.Lambda(lambda x: tf.cast(x, tf.float32))(encoderB_inputs)
xB = layers.Dense(1048, activation="relu", activity_regularizer=l2(l2_lamda))(xB)#
xB = layers.Dense(256, activation="relu",activity_regularizer=l2(l2_lamda))(xB)
xB = layers.Dropout(dropout_rate)(xB)
xB = layers.Dense(64, activation="relu", activity_regularizer=l2(l2_lamda))(xB)
xB = layers.Reshape((n_cells, n_TFs*64), input_shape=(n_cells, n_TFs, 64))(xB)
#xB = layers.Dense(16, activation="relu")(xB)
zB_mean = layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim),activation=None)(xB)
zB =  tfp.layers.MultivariateNormalTriL(latent_dim, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior))(zB_mean)
#encoderB = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
#encoderB.summary()


#Meshing 
y1 = layers.Dense(latent_dim* n_cells, activation="relu", name='meshing1', activity_regularizer=l2(l2_lamda))(zA)
y1 = tf.keras.layers.Reshape((n_TFs, n_cells, latent_dim), input_shape=(n_TFs, n_cells*latent_dim), name='meshing2')(y1)
y1 = tf.transpose(y1, perm=[0, 2, 1, 3])
y2 = layers.Dense(latent_dim* n_TFs, activation="relu", name='meshing3', activity_regularizer=l2(l2_lamda))(zB)
y2 = tf.keras.layers.Reshape((n_cells, n_TFs, latent_dim), input_shape=(n_cells, n_TFs*latent_dim), name='meshing4')(y2)
y = LinearMesh(latent_dim, activation='relu')([y1, y2])
#y=tf.keras.layers.Add()([y1, y2]) #Figrue out how to weight this and add bias -> maybe make own layer

#C, decoder
latent_inputs = y
#xC = layers.Dense(latent_dim*10, activation="relu", name='decoder1')(latent_inputs)
xC = layers.Dense(64, activation="relu", activity_regularizer=l2(l2_lamda))(latent_inputs)#(xC)
xC = layers.Dense(256, activation="relu", name='decoder1', activity_regularizer=l2(l2_lamda))(xC)
xC = layers.Dense(1048, activation="relu", name='decoder2', activity_regularizer=l2(l2_lamda))(xC)
decoderC_outputs = layers.Dense(n_peaks, activation="softmax", name='decoder3', activity_regularizer=l2(l2_lamda))(xC)
#decoderC = keras.Model(latent_inputs, decoder_outputs, name="decoder")
#decoderC.summary()


vae_model = K.Model(inputs=full_inputs, outputs=decoderC_outputs, name="branched_model_2")
vae_model.summary()

###################################################################################
#Training and loss


#vae_model.add_loss(vae_loss)

def neg_log_likelihood(y_true, y_pred):
    y_pred = tfp.distributions.MultivariateNormalTriL(y_pred)
    return -tf.reduce_mean(y_pred.log_prob(y_true))

#negloglik = lambda x, rv_x: -rv_x.log_prob(x)



vae_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=K.losses.MeanSquaredError())

x_train=x_train_6_1000
x_val=x_val_6_1000
x_test=x_test_6_1000

vae_model.fit(x_train, x_train,
            epochs=1000,
            batch_size=12,
            verbose=1,
            validation_data=(x_val, x_val))
###############################################################################
#Viasualizing the data
def visualize(data, labels):
    """
    Input: np.array with shape (m,n)
    """
    if len(data[:,0]) > 50:
        pca = PCA(n_components=30)
        data_pca = pca.fit_transform(data) #Using PCA to decrease dimensionality for tSNE
        data_embedded = TSNE(n_components=2, n_iter=2000, perplexity=10, learning_rate=100).fit_transform(data_pca)
    elif len(data[:, 0]) == 2:
        data_embedded=data
    else:
        data_pca=data
        data_embedded = TSNE(n_components=2, n_iter=2000, perplexity=10, learning_rate=100).fit_transform(data_pca)
    print(data_embedded.shape)

    #y=np.arange(30)
    #color = [item/30 for item in y]
    
    plt.figure(0)
    plt.scatter(data_embedded[:,0], data_embedded[:,1], c=labels, 
                norm=matplotlib.colors.Normalize(), cmap='gist_rainbow')
    plt.title('t-SNE')
    plt.show()




###############################################################################
#Get latent spaces

TF_latent_model = K.Model(inputs=full_inputs, outputs=zA, name='tf_latent')
Cell_latent_model = K.Model(inputs=full_inputs, outputs=zB, name='cell_latent')


TF_latent_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=neg_log_likelihood)

Cell_latent_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=neg_log_likelihood)

tf_latent_test=TF_latent_model.predict(x_train)
print(tf_latent_test.shape)
arr=np.arange(0, 528, step=44)
tf_latent_1=tf_latent_test[arr, :, :].reshape((120, -1))
#tf_splits=tf_latent_test[arr + np.full((44,1),i) for i in range(12), :, :]
#tf_latent_all=np.concatenate(tf_splits, axis=2)
#print(tf_latent_all.shape)
np.save(data_dir+'branched_2_tf_latent_1.npy', tf_latent_test)
print(tf_latent_test)


cell_latent_test=Cell_latent_model.predict(x_train)
cell_latent_1=cell_latent_test[0:44, :, :].reshape((440, -1))
cell_splits=np.split(cell_latent_test, 12, axis=0 )
cell_latent_all=np.concatenate(cell_splits, axis=2)
print(cell_latent_all.shape)
np.save(data_dir+'branched_2_cell_latent_1.npy', cell_latent_test)

visualize(tf_latent_1, np.arange(120))
visualize(cell_latent_1, x_train_6_1000_label)
