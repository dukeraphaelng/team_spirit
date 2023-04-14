from tensorflow.keras.callbacks import ModelCheckpoint
from DAE import DenoisingAutoEncoder
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

class StackedDenoisingAutoEncoder():
    def __init__(self, layers, corruption, activate_encoder, activate_decoder):
        self.layers = layers
        self.corruption = corruption
        self.activate_encoder = activate_encoder
        self.activate_decoder = activate_decoder
        self.encoding_funcs = []
        self.encoder_layers = []
        self.history = None
  
    def pretrain(self, X_train, X_val, epochs, batch_size):
        # self.layers contains the units each denoising autoencoder should take in
        learnt_input = X_train
        encoded_validation = X_val
        for layer in self.layers:
            autoencoder = DenoisingAutoEncoder([layer], self.corruption, self.activate_encoder, self.activate_decoder)
            (encoding_function, encoder) = autoencoder.forward(learnt_input, encoded_validation, epochs, batch_size)
            learnt_input = encoding_function.predict(learnt_input)
            encoded_validation = encoding_function.predict(encoded_validation)

            self.encoding_funcs.append(encoding_function)
            self.encoder_layers.append(encoder)

        self.history = autoencoder.history

    def finetune(self, X_train, Y_train, X_val, Y_val, epochs, batch_size):
        encoder_input = Input(shape = (X_train.shape[1], ))

        final_encoder = encoder_input
        for encoder in self.encoder_layers:
            final_encoder = encoder(final_encoder)

        # Define the logistic regression layer
        lr_layer = Dense(Y_train.shape[1], activation='softmax')
        predictions = lr_layer(final_encoder)

        # Create the fine-tuned model
        fine_tuned_model = Model(inputs=encoder_input, outputs=predictions)
        fine_tuned_model.compile(loss='categorical_crossentropy', optimizer='adam')

        # Select best model
        checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
        self.history = fine_tuned_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), callbacks=[checkpointer])
        fine_tuned_model.load_weights('weights.hdf5')

        fine_tuned_encoder = Model(inputs=encoder_input, outputs=final_encoder)
        return (fine_tuned_model, fine_tuned_encoder)

    def encode(self, X):
        encoded_representation = X
        for func in self.encoder_layers:
            encoded_representation = func.predict(encoded_representation)
        return encoded_representation