import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import matplotlib.pyplot as plt

# Load the credit card fraud dataset
trainData = pd.read_csv("./Dataset/trainingData").astype("float32")

# shuffle dataset
trainData = trainData.sample(frac=1, random_state=42)

# Normalize the data
trainData = (trainData - trainData.mean()) / trainData.std()

# Input shape for credit card transactions
input_shape = trainData.shape[1:]

# hyperparameters
latent_dim = 2
epochs = 50
batch_size = 128

tfd = tfp.distributions

prior = tfd.MultivariateNormalDiag(
        loc=tf.zeros(latent_dim))

encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape, name='encoder_input'),
    tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(8, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim), activation=None),
    tfp.layers.MultivariateNormalTriL(latent_dim, 
                           activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior)),
], name='encoder')

decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[latent_dim]),
    tf.keras.layers.Dense(8, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(input_shape), activation="linear"), 
    tfp.layers.IndependentNormal(input_shape),
], name='decoder')

vae = tf.keras.Model(inputs=encoder.inputs,
                outputs=decoder(encoder.outputs[0]))

def negative_log_likelihood(x, rv_x):
    return -tf.reduce_sum(rv_x.log_prob(x), axis=-1)

vae.compile(optimizer="adam", 
            loss=negative_log_likelihood)

# training callbacks
terminationProtocol = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
)

# Train the VAE
model = vae.fit(x=trainData, y=trainData,
        batch_size=batch_size, epochs=epochs, 
        validation_split=0.2, 
        callbacks=[terminationProtocol])


plt.plot(model.history['loss'], label="Training loss")
plt.plot(model.history['val_loss'], label="Validation loss")
plt.legend(loc='upper right')
plt.show()