import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K
from sklearn.decomposition import PCA

np.random.seed(0)
tf.random.set_seed(0)

all_data = tf.cast(tf.convert_to_tensor(np.load('allProbPCA2.npy')), tf.float32) # TF x CELL x WINDOW
n_TFs, n_cells, n_vars = all_data.shape 

num_epochs = 10 
batch_size = 100
learning_rate = 1e-5 

randtf, randcell = np.unravel_index(np.random.permutation(n_TFs * n_cells), (n_TFs, n_cells))
tf_data = tf.reshape(all_data, shape=(n_TFs, n_cells * n_vars))
cell_data = tf.reshape(tf.transpose(all_data, perm=[1, 0, 2]), shape=(n_cells, n_TFs * n_vars))

def make_batch(arr_tf, arr_cell):
    x = tf.stack([tf_data[item,:] for item in arr_tf])
    y = tf.stack([cell_data[item,:] for item in arr_cell])
    target = tf.stack([all_data[arr_tf[i], arr_cell[i], :] for i in range(batch_size)])
    return x, y, target

class Encoder(K.layers.Layer):
    def __init__(self, interm_dim=2000, interm_dim2 = 100, latent_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.encoder1 = K.layers.Dense(interm_dim, activation='relu')
        self.encoder2 = K.layers.Dense(interm_dim2, activation='relu')
        self.avg = K.layers.Dense(latent_dim, activation='relu')
        self.logsigma = K.layers.Dense(latent_dim, activation=None)    

    def call(self, inputs):
        x0 = self.encoder1(inputs)
        x = self.encoder2(x0)
        zbar = self.avg(x)
        logsigma = self.logsigma(x)
        return zbar, logsigma

class Decoder(K.layers.Layer):
    def __init__(self, interm_dim=200, output_dim=n_vars, **kwargs):
        super().__init__(**kwargs)
        self.decoder1 = K.layers.Dense(interm_dim, activation='relu')
        self.decoder2 = K.layers.Dense(interm_dim, activation='sigmoid')
        self.decoder3 = K.layers.Dense(output_dim, activation=None)
    
    def call(self, inputs):
        x = self.decoder1(inputs)
        y = self.decoder2(x)
        z = self.decoder3(y)
        return z

class VAE2(K.Model):
    def __init__(self, encoder_interm_dims=2000, encoder_interm_dims_2=100, latent_dims=20, decoder_interm_dims=50, output_dims=n_vars, **kwargs):
        super().__init__(**kwargs)
        self.latent_dims = latent_dims
        self.encoder1 = Encoder(encoder_interm_dims, encoder_interm_dims_2, latent_dims)
        self.encoder2 = Encoder(encoder_interm_dims, encoder_interm_dims_2, latent_dims)
        self.decoder = Decoder(decoder_interm_dims, output_dims)

    def call(self, inputs):
        x, y = inputs
        zbar1, ls1 = self.encoder1(x)
        zbar2, ls2 = self.encoder2(y)
        zbar = tf.concat([zbar1, zbar2], axis=1)
        ls = tf.concat([ls1, ls2], axis=1)
        kl_loss = - 0.5 * tf.reduce_mean(ls - tf.square(zbar) - tf.exp(ls) + 1, axis=1)
        self.add_loss(kl_loss)
        epsilon = K.backend.random_normal(shape=(batch_size, 2*self.latent_dims))
        z = zbar + tf.exp(0.5 * ls) * epsilon
        return zbar, self.decoder(z)

model = VAE2()
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
        loss += model.losses
    grad = tape.gradient(loss, model.trainable_weights)
    opt.apply_gradients(zip(grad, model.trainable_weights))
    
    return tf.math.reduce_mean(loss)

def compute_val_loss(epoch):
    val_loss = 0.0
    for i in range(num_train, num_train + num_val):
         x, y, target = make_batch(randtf[i*batch_size:(i+1)*batch_size], randcell[i*batch_size:(i+1)*batch_size])
         _, predictions = model((x, y), training=False)
         val_loss += loss_fn(predictions, target) 
    print("Epoch {:d}, validation loss: {:.6E}".format(epoch, tf.math.reduce_sum(val_loss/num_val)))

num_train = 500 # 1200 for vae latent
num_val = 50 # 100 for vae latent
num_test = 40 # 80 for vae latent 
save_dir = "ae_weights_pca/"

for _ in range(num_epochs):
    for i in range(num_train):
         x, y, target = make_batch(randtf[i*batch_size:(i+1)*batch_size], randcell[i*batch_size:(i+1)*batch_size])
         current_loss = train_step((x,y), target)
         if i%50 == 0:
             print("Epoch {:d}, batch {:d}, batch loss: {:.6E}".format(_+1, i, current_loss))
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

np.save('VAETFLatentLowE.npy', tfLatent)
np.save('VAECellLatentLowE.npy', cellLatent)
