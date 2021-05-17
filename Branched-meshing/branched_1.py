# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:37:20 2021

@author: clair
"""

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

#import skimage.measure

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

training_set = None #should have shape [n_TFs, n_cells, n_peaks]
training_labels = None
test_set = None
test_labels = None

#hyperparams, will tune eventually
n_TFs=10
n_peaks=10045  #173978 #check this for the dataset using
batch_size=12
n_cells=10
latent_dim=4
dropout_rate=.5
l2_lamda=.01
learning_rate=0.001

#For consistency let's check the python version
print('Python version: {}'.format(platform.sys.version))

# Let's also check the Tensorflow version
tf_version = tf.__version__
print('TensorFlow version: {}'.format(tf_version))
#########################################################################
#Data loading
data_dir='/nobackup/users/btu/shared/dlproj/cstg/'

x_train_6_1000=np.load(data_dir+'x_train_6_cells_1000_pool.npy', allow_pickle=True)
x_val_6_1000=np.load(data_dir+'x_val_6_cells_1000_pool.npy', allow_pickle=True)
x_test_6_1000=np.load(data_dir+'x_test_6_cells_1000_pool.npy', allow_pickle=True)

x_train_6_1000_label=np.load(data_dir+'x_train_6_cells_1000_pool_labels.npy', allow_pickle=True)
x_val_6_1000_label=np.load(data_dir+'x_val_6_cells_1000_pool_labels.npy', allow_pickle=True)
x_test_6_1000Lable=np.load(data_dir+'x_test_6_cells_1000_pool_labels.npy', allow_pickle=True)


#########################################################################
#Some layers and functions

#Sampling layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        latent=tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim, latent))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    
#Viasualizing the data
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
#########################################################################
#Model

#For a branched model want to use the keras functional api (instead of sequential)



full_inputs = K.Input(shape=(n_cells, n_TFs, n_peaks))



#Layers

#A branch encoder for TF latent space
encoderA_inputs = tf.transpose(full_inputs, perm=[0, 2, 1, 3])
xA = layers.Conv2D(64, 2, activation="relu", strides=1, padding="same", activity_regularizer=l2(l2_lamda))(encoderA_inputs)
xA = layers.Conv2D(32, 10, activation="relu", strides=1, padding="same", activity_regularizer=l2(l2_lamda))(xA)
xA = layers.Dropout(dropout_rate)(xA)
xA = layers.Conv2D(16, 10, activation="relu", strides=1, padding="same", activity_regularizer=l2(l2_lamda))(xA)
xA = layers.Reshape((n_TFs, n_cells*16), input_shape=(n_TFs, n_cells, 16), activity_regularizer=l2(l2_lamda))(xA)
#xA = layers.Dense(16, activation="relu")(xA)
zA_mean = layers.Dense(latent_dim, name="zA_mean", activity_regularizer=l2(l2_lamda))(xA)
zA_log_var = layers.Dense(latent_dim, name="zA_log_var", activity_regularizer=l2(l2_lamda))(xA)
zA = Sampling()([zA_mean, zA_log_var])
#encoderA = keras.Model(encoder_inputs, [zA_mean, zA_log_var, zA], name="encoder")
#encoderB.summary()

#B branch encoder for cell latent space
encoderB_inputs=full_inputs
xB = layers.Conv2D(64, 2, activation="relu", strides=1, padding="same", activity_regularizer=l2(l2_lamda))(encoderB_inputs)#
xB = layers.Conv2D(32, 10, activation="relu", strides=1, padding="same", activity_regularizer=l2(l2_lamda))(xB)
xB = layers.Conv2D(16, 10, activation="relu", strides=1, padding="same", activity_regularizer=l2(l2_lamda))(xB)
xB = layers.Dropout(dropout_rate)(xB)
xB = layers.Reshape((n_cells, n_TFs*16), input_shape=(n_cells, n_TFs, 16))(xB)
#xB = layers.Dense(16, activation="relu")(xB)
zB_mean = layers.Dense(latent_dim, name="zB_mean", activity_regularizer=l2(l2_lamda))(xB)
zB_log_var = layers.Dense(latent_dim, name="zB_log_var", activity_regularizer=l2(l2_lamda))(xB)
zB = Sampling()([zB_mean, zB_log_var])
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
xC = layers.Conv2DTranspose(16, 10, activation="relu", strides=1, padding="same", activity_regularizer=l2(l2_lamda))(latent_inputs)#(xC)
xC = layers.Conv2DTranspose(64, 10, activation="relu", strides=1, padding="same", name='decoder1', activity_regularizer=l2(l2_lamda))(xC)
xC = layers.Conv2DTranspose(128, 2, activation="relu", strides=1, padding="same", name='decoder2', activity_regularizer=l2(l2_lamda))(xC)
decoderC_outputs = layers.Dense(n_peaks, activation="sigmoid", name='decoder3', activity_regularizer=l2(l2_lamda))(xC)
#decoderC = keras.Model(latent_inputs, decoder_outputs, name="decoder")
#decoderC.summary()


vae_model = K.Model(inputs=full_inputs, outputs=decoderC_outputs, name="branched_model")
vae_model.summary()
###################################################################

#Training and loss
def vae_loss_fn(inputs, outputs):
    #cce=K.losses.CategoricalCrossentropy()
    mse = K.losses.MeanSquaredError()
    reconstruction_loss=mse(inputs, outputs)
    #reconstruction_loss *=1
    kl_A_loss = 1 + zA_log_var - K.backend.square(zA_mean) - K.backend.exp(zA_log_var)
    kl_B_loss = 1 + zB_log_var - K.backend.square(zB_mean) - K.backend.exp(zB_log_var)
    kl_A_loss = K.backend.sum(kl_A_loss, axis=-1)*-0.5
    kl_B_loss = K.backend.sum(kl_B_loss, axis=-1)*-0.5
    vae_loss = K.backend.mean(reconstruction_loss + kl_A_loss+kl_B_loss)
    return vae_loss

#vae_model.add_loss(vae_loss)


vae_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=vae_loss_fn)

x_train=x_train_6_1000
x_val=x_val_6_1000
x_test=x_test_6_1000

vae_model.fit(x_train, x_train,
            epochs=100,
            batch_size=12,
            verbose=1,
            validation_data=(x_val, x_val))

#############################################################################

#Get latent spaces

#Get latent spaces
print(x_train_6_1000_label.shape)

TF_latent_model = K.Model(inputs=full_inputs, outputs=zA, name='tf_latent')
Cell_latent_model = K.Model(inputs=full_inputs, outputs=zB, name='cell_latent')


TF_latent_model.compile()

Cell_latent_model.compile()

tf_latent_test=TF_latent_model.predict(x_train)
tf_latent_1=tf_latent_test[0:12, :, :]
tf_latent_1=tf_latent_1.reshape((120, -1))
np.save(data_dir+'branched_1_tf_latent_1.npy', tf_latent_test)



cell_latent_test=Cell_latent_model.predict(x_train)
arr=np.arange(0, 528, step=12)
cell_latent_split=np.split(cell_latent_test, 528, axis=0)
cell_latent_1=cell_latent_test[44:88, :, :]
split=np.split(cell_latent_1, 44, axis=0)
cell_latent_1=np.squeeze(np.concatenate(split, axis=1))
print(cell_latent_1.shape)
np.save(data_dir+'branched_1_cell_latent_1.npy', cell_latent_test)
cell_latent_1=cell_latent_1.reshape((440, -1))

visualize(tf_latent_1, np.arange(120))
visualize(cell_latent_1, x_train_6_1000_label)

###############################################################################
#Comparing input to output
cell_1_train=x_train[0:44, :, :, :].reshape((440, -1))
x_pred=vae_model.predict(x_train)
cell_1_pred=x_pred[0:44, :, :, :].reshape((440, -1))
visualize(cell_1_train, x_train_6_1000_label)
visualize(cell_1_pred, x_train_6_1000_label)
