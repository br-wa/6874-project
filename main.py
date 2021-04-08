import os
import platform
import random
import shutil
import sys
import functools
import itertools as it
import copy

import numpy as np
import pandas as pd
import sklearn.metrics
import h5py

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as K

n_TFs = 0
n_cells = 0
n_peaks = 0

training_set = None #should have shape [n_TFs, n_cells, n_peaks]
training_labels = None
test_set = None
test_labels = None

#hyperparams, will tune eventually
num_epochs = 25
batch_size = 100
learning_rate = 0.005

def get_data():
    #TODO load and premute data into desired shape  
    data_dir = os.path.join('.', None) #TODO

def label_data():
    #some silly "dummy" label 

def initialize_factorCompactor():
    #defining A as K.Sequential()
    A = K.Sequential()
    A.add(
        K.layers.Dense(
            input_shape=(None, n_cells * n_peaks)
            units=4,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeroes'
        )
    )
    A.add(
        K.layers.Dense(
            units=2,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeroes'
        )
    )
    return A

def initialize_cellCompactor():
    #defining B as K.Sequential(), just copying A for now
    B = K.Sequential()
    B.add(
        K.layers.Dense(
            input_shape=(None, n_TFs * n_peaks)
            units=4,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeroes'
        ) 
    )
    B.add(
        K.layers.Dense(
            units=2,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeroes'
        )
    )
    return B

def initialize_extractor():
    #defining C as K.Sequential()

    return C

def loss(predictions, y):
    #defining loss function, probably just MSE for now?

def train_step(factorCompactor, cellCompactor, extractor, x, y, optimizer):
    with tf.gradientTape() as tape:
        predictions = extractor(
            tf.concat(
                factorCompactor(x[:,0:n_cells*n_peaks]), 
                cellCompactor(x[:,n_cells*n_peaks:])), 
                axis=0
            )
        )
        pass_loss = loss(predictions, y)
    gradients = tape.gradient(pass_loss, factorCompactor.trainable_variables+cellCompactor.trainable_variables+extractor.trainable_variables)
    optimizer.apply_gradients(zip(gradients, factorCompactor.trainable_variables+cellCompactor.trainable_variables+extractor.trainable_variables))
    
    return pass_loss

def train(factorCompactor, cellCompactor, extractor):
    optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
    for _ in range(num_epochs):
        


def main():
    get_data()
    label_data()
    factorCompactor = initialize_factorCompactor()
    cellCompactor = initialize_cellCompactor()
    extractor = initialize_extractor()
    train(factorCompactor, cellCompactor, extractor)
