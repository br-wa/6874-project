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

training_set = None #should have shape [n_TFs, n_cells, n_peaks]
training_labels = None
test_set = None
test_labels = None

#hyperparams, will tune eventually
num_epochs = 25
batch_size = 100
learning_rate = 0.005

def get_data():
    #TODO load and process data  
    data_dir = os.path.join('.', None) #TODO

def label_data():
    #some silly "dummy" label

def initialize_A():
    #defining A as K.sequential()

    return A

def initialize_B():
    #defining B as K.sequential()

    return B

def initialize_C():
    #defining C as K.sequential()

    return C

def loss(predictions, y):
    #defining loss function, probably just MSE for now?

def train_step(A, B, C, x, y, optimizer):
    with tf.gradientTape() as tape:
        predictions = C(tf.concat(A(x[:,0:blah]), B(x[:,blah+1:])), axis=0) #TODO set blah
        pass_loss = loss(predictions, y)
    gradients = tape.gradient(pass_loss, A.trainable_variables+B.trainable_variables+C.trainable_variables)
    optimizer.apply_gradients(zip(gradients, A.trainable_variables+B.trainable_variables+C.trainable_variables))
    
    return pass_loss

def train(A, B, C):
    optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
    for _ in range(num_epochs):
        


def main():
    get_data()
    A = initialize_A()
    B = initialize_B()
    C = initialize_C()
    
