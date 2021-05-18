import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K
from sklearn.decomposition import PCA

np.random.seed(5)
tf.random.set_seed(5)

all_data = tf.cast(tf.convert_to_tensor(np.load('hk_latent.npy')), tf.float32) # TF x CELL x WINDOW
n_TFs, n_cells, n_vars = all_data.shape 

num_epochs = 20 
batch_size = 100
learning_rate = 1e-6 

randtf, randcell = np.unravel_index(np.random.permutation(n_TFs * n_cells), (n_TFs, n_cells))
tf_data = tf.reshape(all_data, shape=(n_TFs, n_cells * n_vars))
cell_data = tf.reshape(tf.transpose(all_data, perm=[1, 0, 2]), shape=(n_cells, n_TFs * n_vars))

def make_batch(arr_tf, arr_cell):
    x = tf.stack([tf_data[item,:] for item in arr_tf])
    y = tf.stack([cell_data[item,:] for item in arr_cell])
    target = tf.stack([all_data[arr_tf[i], arr_cell[i], :] for i in range(batch_size)])
    return x, y, target

class Encoder(K.layers.Layer):
    def __init__(self, interm_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.encoder1 = K.layers.Dense(interm_dim, activation='relu')
        self.encoder2 = K.layers.Dense(latent_dim, activation='relu')
    
    def call(self, inputs):
        x = self.encoder1(inputs)
        z = self.encoder2(x)
        return z

class Decoder(K.layers.Layer):
    def __init__(self, interm_dim, output_dim=n_vars, **kwargs):
        super().__init__(**kwargs)
        self.decoder1 = K.layers.Dense(interm_dim, activation='relu')
        self.decoder2 = K.layers.Dense(output_dim, activation='sigmoid')
    
    def call(self, inputs):
        x = self.decoder1(inputs)
        z = self.decoder2(x)
        return z

class AE2(K.Model):
    def __init__(self, encoder_interm_dims=1000, latent_dims=20, decoder_interm_dims=60, output_dims=n_vars, **kwargs):
        super().__init__(**kwargs)
        self.encoder1 = Encoder(encoder_interm_dims, latent_dims)
        self.encoder2 = Encoder(encoder_interm_dims, latent_dims)
        self.decoder = Decoder(decoder_interm_dims, output_dims)

    def call(self, inputs):
        x, y = inputs
        z = tf.concat([self.encoder1(x), self.encoder2(y)], axis=1) # axis 0 is batch
        return z, self.decoder(z)

model = AE2()
opt = K.optimizers.Adam(learning_rate)
loss_fn = K.losses.MeanAbsoluteError()

@tf.function(
    input_signature=(
        (
            tf.TensorSpec((batch_size, n_cells * n_vars)), 
            tf.TensorSpec((batch_size, n_TFs * n_vars))
        ), 
        tf.TensorSpec((batch_size, n_vars))
    )
)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        _, predictions = model(inputs, training=True)
        loss = loss_fn(predictions, target)
    grad = tape.gradient(loss, model.trainable_weights)
    opt.apply_gradients(zip(grad, model.trainable_weights))
    
    return tf.math.reduce_mean(loss)

def compute_val_loss(epoch):
    val_loss = 0.0
    for i in range(num_train, num_train + num_val):
         x, y, target = make_batch(randtf[i*batch_size:(i+1)*batch_size], randcell[i*batch_size:(i+1)*batch_size])
         _, predictions = model((x, y), training=False)
         val_loss += loss_fn(predictions, target)
    print("Epoch {:d}, validation loss: {:.2E}".format(epoch, val_loss))

num_train = 500
num_val = 50
num_test = 40
save_dir = "ae_weights/"

for _ in range(num_epochs):
    for i in range(num_train):
         x, y, target = make_batch(randtf[i*batch_size:(i+1)*batch_size], randcell[i*batch_size:(i+1)*batch_size])
         current_loss = train_step((x,y), target)
         if i%50 == 0:
             print("Epoch {:d}, iteration {:d}, average loss: {:.2E}".format(_+1, i, current_loss))
    compute_val_loss(_+1)
    save_path = save_dir + "model" + str(_)
    model.save_weights(save_path)

tfLatent = []
cellLatent = []
for i in range(max(n_TFs, n_cells)):
    z, _ = model((tf.expand_dims(tf_data[i%n_TFs], axis=0), tf.expand_dims(cell_data[i%n_cells],axis=0)), training=False)
    if i < n_TFs:
        tfLatent.append(z[0,:20])
    if i < n_cells:
        cellLatent.append(z[0,20:])

tfLatent = (tf.stack(tfLatent)).numpy()
cellLatent = (tf.stack(cellLatent)).numpy()

np.save('NewAETFLatent.npy', tfLatent)
np.save('NewAECellLatent.npy', cellLatent)
