import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd

# Load the credit card fraud dataset
trainData = pd.read_csv("./Dataset/trainingData").astype("float32")

# shuffle dataset
trainData = trainData.sample(frac=1, random_state=42)

# Normalize the data
trainData = (trainData - trainData.mean()) / trainData.std()

# Class function for VAE
class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, inputShape, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.inputShape = inputShape
        self.latent_dim = latent_dim
        self.prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(latent_dim))
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.inputShape, name='encoder_input'),
            tf.keras.layers.Dense(20, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(10, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(8, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(4, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(self.latent_dim), activation=None),
            tfp.layers.MultivariateNormalTriL(self.latent_dim, 
                                               activity_regularizer=tfp.layers.KLDivergenceRegularizer(self.prior)),
        ], name='encoder')
        return encoder

    def build_decoder(self):
        decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=[self.latent_dim]),
            tf.keras.layers.Dense(4,activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(8, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(10, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(20, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(self.inputShape[0], activation=tf.nn.sigmoid), 
        ], name='decoder')
        return decoder
    
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        batch_size = tf.shape(mean)[0]
        eps = tf.random.normal(shape=(batch_size, self.latent_dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        decoded = self.decode(z)
        return decoded

def loss_function(y_true, y_pred):
    z_logvar, z_mean = vae.encode(y_true)
    kl_divergence_loss = -0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=-1)
    reconstruction_loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1) 
    loss = tf.reduce_mean(reconstruction_loss + kl_divergence_loss)
    return loss

# Create an instance of the VAE model
inputShape = trainData.shape[1:]
latent_dim = 2
vae = VariationalAutoencoder(inputShape, latent_dim)

# Compile the model
vae.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss_function)

# Model callbacks
terminationProtocol = tf.keras.callbacks.EarlyStopping(
    monitor='loss', 
    patience=8, 
    restore_best_weights=True,
)

bestWeights = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights.keras',
    verbose=0,
    save_weights_only=True
)

# Train the VAE
model = vae.fit(x=trainData, y=trainData,
        batch_size=128, epochs=50, 
    callbacks=[bestWeights, terminationProtocol])
