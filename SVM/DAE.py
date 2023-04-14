from DenseTranpose import DenseTranspose
import numpy as np
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

# Class for construction of a denoising autoencoder
np.random.seed(55)
class DenoisingAutoEncoder:
    def __init__(self, layers, corruption, activate_encoder, activate_decoder):
        self.layers = layers
        self.corruption = corruption
        self.activate_encoder = activate_encoder
        self.activate_decoder = activate_decoder
        self.history = None
  
    def forward(self, X_train, X_val, epochs, batch_size):
        # Step 1, Add binomial noise
        X_train_noisy = self.inject_noise(X_train)

        # Step 2, Encode X_train_noisy using sigmoid
        encoder_input = Input(shape = (X_train_noisy.shape[1], ))
        encoder = Dense(self.layers[0], activation=self.activate_encoder)
        final_encoder = encoder(encoder_input)

        # Step 3, Decode X_train_noisy using sigmoid
        # Tie the weights between the encoder and decoder layers
        decoder = DenseTranspose(encoder, activation=self.activate_decoder)
        final_decoder = decoder(final_encoder)

        # Step 4, cross entropy loss for normalised data and adam optimizer (Not sure what optimizer the paper uses)
        autoencoder = Model(encoder_input, final_decoder)
        autoencoder.compile(loss = 'binary_crossentropy', optimizer = 'adam')

        # Train it
        self.history = autoencoder.fit(X_train_noisy, X_train, batch_size = batch_size, epochs = epochs, validation_data=(X_val, X_val))
        autoencoder.summary()

        # Get the model that maps input to its encoded representation
        encoder_model = Model(encoder_input, final_encoder)

        # Return the (encoding model, encoding function)
        return (encoder_model, encoder)

    def inject_noise(self, x):
        # inject binomial noise since this model assumes you are normalising input 
        # with min max normalisation
        mask = np.random.choice([0, 1], size=x.shape, p=[self.corruption, 1-self.corruption])
        X_noisy = x * mask
        return X_noisy