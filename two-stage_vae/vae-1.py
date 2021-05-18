#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 13:56:23 2021

@author: btu
"""


import numpy as np
import sys
import os 
if os.path.exists(f'{sys.argv[1]}.npy'):
    sys.exit()
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import shutil
#import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import glob
import dill
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

with open(sys.argv[1],'rb') as f:
    b=dill.load(f)
b=tf.sparse.to_dense(b)
b=tf.reshape(b,[b.shape[0],1,b.shape[1],1])
b=tf.image.resize(b,[1,500000])
b=tf.squeeze(b)
print(b)
#b=b[:,:100000]
dataset=tf.data.Dataset.from_tensor_slices((b,b)).batch(64)
dataset=dataset.shuffle(10000)
train_dataset=dataset.skip(10).take(-1)
eval_dataset=dataset.take(10)



#%%


input_shape = b.shape[-1]
encoded_size =128 

prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
			reinterpreted_batch_ndims=1)
encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape),
    
    tfkl.Lambda(lambda x: tf.cast(x, tf.float32)),

    tfkl.Dense(1024,activation='relu'),

    tfkl.Dense(256,activation='relu'),
    tfkl.Dense(128,activation='relu',kernel_regularizer='l2'),
    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
	       activation=None),
    tfpl.MultivariateNormalTriL(
	encoded_size,
	activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
])
decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[encoded_size]),
    tfkl.Dense(128,activation='relu'),
    tfkl.Dense(256,activation='relu'),

    tfkl.Dense(1024,activation=None),

    tfkl.Dense(tfpl.IndependentBernoulli.params_size(input_shape)),
    tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
   
])

vae = tfk.Model(inputs=encoder.inputs,
		    outputs=decoder(encoder.outputs[0]))
    
negloglik = lambda x, rv_x: -rv_x.log_prob(x)
    
checkpoint_filepath = f"/dev/shm/{sys.argv[1].split('.')[0]}.checkpoint"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
filepath=checkpoint_filepath,
save_weights_only=True,
monitor='val_loss',
mode='min',
save_best_only=True)

vae.compile(optimizer=tf.optimizers.Adam(learning_rate=5e-4),
	    loss=negloglik)

vae.fit(train_dataset,
		    epochs=20,
             		validation_data=eval_dataset,callbacks=[model_checkpoint_callback])
vae.load_weights(checkpoint_filepath)
encoder.compile()
z=encoder.predict_on_batch(b)
np.save(sys.argv[1].split('.')[0],z)
shutil.rmtree('/dev/shm/*',ignore_errors=1)
