import os
import numpy as np
import keras
import joblib
import pickle

class svm_model():
    def __init__(self, dataset):

        if dataset == 'bert':
            first_path = "svm_bert_models"
        else:
            first_path = "svm_models"
            
        path_to_encoder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            first_path,
            "sdae_encoder.h5"
        )

        path_to_svm = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
           first_path,
            "svm_model.pki"
        )

        path_to_scaler = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            first_path,
            "min_max_scaler.pkl"
        )

        path_to_labeller = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            first_path,
            "label_encoder.pkl"
        )

        path_to_vectorizer = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            first_path,
            "count_vectorizer.pkl"
        )
        self.encoder = keras.models.load_model(path_to_encoder, compile=False)
        self.encoder.compile(loss='categorical_crossentropy', optimizer='adam')
        self.svm = joblib.load(path_to_svm)
        with open(path_to_scaler, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(path_to_labeller, 'rb') as f:
            self.label = pickle.load(f)
        with open(path_to_vectorizer, 'rb') as f:
            self.count_vectorizer = pickle.load(f)

        params = (np.size(self.svm.coef_))
        print(params)

    def predict_name(self, x):
        encoded_x = self.count_vectorizer.transform(x).toarray()
        encoded_x = self.scaler.transform(encoded_x)
        encoded_x = self.encoder.predict(encoded_x)
        probability = self.svm.predict(encoded_x)
        return (self.label.inverse_transform(probability))
    
    def predict(self, x):
        encoded_x = self.count_vectorizer.transform(x).toarray()
        encoded_x = self.scaler.transform(encoded_x)
        encoded_x = self.encoder.predict(encoded_x)
        probability = self.svm.predict(encoded_x)
        return probability

    def predict_proba(self, x):
        encoded_x = self.count_vectorizer.transform(x).toarray()
        encoded_x = self.scaler.transform(encoded_x)
        encoded_x = self.encoder.predict(encoded_x)
        probability_distribution = self.svm.predict_proba(encoded_x)
        return probability_distribution
    
    def label_data(self, x):
        return self.label.transform(x)