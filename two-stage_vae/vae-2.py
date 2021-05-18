#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 13:56:23 2021

@author: btu
"""


import numpy as np
import sys
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import shutil

import tensorflow_probability as tfp
import glob
import dill
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


assert len(sys.argv)==3

#%%




zList=[]
for z in glob.glob(sys.argv[1]+'/'+'*.npy'):
	zList.append(np.load(z).flatten()[:,np.newaxis])
print(zList[0].shape)
b=tf.concat(zList,1)
b=tf.transpose(b)
print(b.shape)
print(b)

dataset=tf.data.Dataset.from_tensor_slices((b,b)).batch(1)
dataset=dataset.shuffle(800)
train_dataset=dataset.skip(10).take(-1)
eval_dataset=dataset.take(10)



#%%

input_shape = b.shape[-1]
encoded_size = 16

prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
			reinterpreted_batch_ndims=1)
encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape),
    
    tfkl.Lambda(lambda x: tf.cast(x, tf.float32)),

    tfkl.Dense(1024,activation='relu'),

    tfkl.Dense(512,activation='relu'),

    tfkl.Dense(256,activation='relu'),
    
    tfkl.Dense(128,activation='relu'),
    tfkl.Dense(64,activation='relu'),
    tfkl.Dense(32,activation='relu'),
    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
	       activation=None),
    tfpl.MultivariateNormalTriL(
	encoded_size,
	activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
])
decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[encoded_size]),
    tfkl.Dense(32,activation='relu'),

    tfkl.Dense(64,activation='relu'),
    tfkl.Dense(128,activation='relu'),
    tfkl.Dense(256,activation='relu'),
    tfkl.Dense(512,activation='relu'),
    tfkl.Dense(1024,activation='relu'),
    tfkl.Dense(input_shape,activation=None),
 
])

vae = tfk.Model(inputs=encoder.inputs,
		    outputs=decoder(encoder.outputs[0]))
    
negloglik = lambda x, rv_x: -rv_x.log_prob(x)
    
checkpoint_filepath = f"{sys.argv[2]}.checkpoint"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
filepath=checkpoint_filepath,
save_weights_only=True,
monitor='val_loss',
mode='min',
save_best_only=True)

vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),
		loss=tf.losses.MeanSquaredError())

vae.fit(train_dataset,
		    epochs=1500,
             		validation_data=eval_dataset,callbacks=[model_checkpoint_callback])
vae.load_weights(checkpoint_filepath)
encoder.compile()
z=encoder.predict_on_batch(b)
np.save(f'{sys.argv[2]}',z)
