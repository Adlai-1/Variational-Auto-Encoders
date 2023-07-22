import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the credit card fraud dataset
# Make sure you provide the correct path to the CSV file.
dataset_path = './archive/creditcard.csv'
Dataset = pd.read_csv(dataset_path).astype('float32')

# Separate fraud and non-fraud data
fraudData = Dataset[Dataset['Class'] == 1]
nonfraudData = Dataset[Dataset['Class'] == 0]

# Drop 'Class' column from both datasets
fraudData = fraudData.drop(['Class'], axis=1)
nonfraudData = nonfraudData.drop(['Class'], axis=1)

# Split nonfraudData into training and testing datasets
trainData, testData = train_test_split(nonfraudData, train_size=0.7, test_size=0.3, random_state=42)

# Split testData into training and validation
testData, valData = train_test_split(testData, train_size=0.5, test_size=0.5, random_state=42)

# Normalize the data
trainData = (trainData - trainData.mean()) / trainData.std()
valData = (valData - valData.mean()) / valData.std()
testData = (testData - testData.mean()) / testData.std()

# Input shape for credit card transactions
input_shape = trainData.shape[1:]

# VAE Hyperparameters
latent_dim = 4
epochs = 30
batch_size = 32

# Define the VAE model
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_input = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Dense(128, activation='relu')(encoder_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Encoder outputs
        z_mean = tf.keras.layers.Dense(self.latent_dim)(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim)(x)

        return tf.keras.models.Model(encoder_input, [z_mean, z_log_var])

    def build_decoder(self):
        decoder_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(decoder_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Decoder output
        output = tf.keras.layers.Dense(input_shape[0], activation='sigmoid')(x)

        return tf.keras.models.Model(decoder_input, output)

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        return self.decoder(z)

# Define the VAE loss function
def vae_loss(y_true, y_pred):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
    kl_loss = -0.5 * tf.reduce_sum(1 + vae.z_log_var - tf.square(vae.z_mean) - tf.exp(vae.z_log_var), axis=-1)
    return reconstruction_loss + kl_loss

# Instantiate and compile the VAE model
vae = VAE(latent_dim)
vae.compile(optimizer='adam', loss=vae_loss)

# training callbacks
terminationProtocol = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', 
    patience=15, 
    restore_best_weights=True,
    min_delta=0.05
)

bestWeights = tf.keras.callbacks.ModelCheckpoint(
    filepath='model3.tf',
    verbose=0,
    save_best_only=True,
)

# Train the VAE
vae.fit(trainData, trainData, 
        batch_size=batch_size, epochs=epochs, 
        validation_data=(valData, valData), 
        callbacks=[terminationProtocol, bestWeights])