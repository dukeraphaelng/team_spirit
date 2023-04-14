# %% [markdown]
# # Support Vector Machine with SDAE using sklearn SVM

# Imports
import os
import numpy as np
import pandas as pd

# Classifier
from sklearn.svm import SVC

# Character N-gram feature extractor
from sklearn.feature_extraction.text import CountVectorizer

# Util
from data_io import get_book
import torch
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Keras 
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

# Plotting
import matplotlib.pyplot as plt

# SDAE
from SDAE import StackedDenoisingAutoEncoder

import tensorflow as tf

# Create the training, test and validation sets
train_data = pd.concat(pd.read_csv("train.csv", chunksize = 100))
test_data = pd.concat(pd.read_csv("test.csv", chunksize = 100))
val_data = pd.concat(pd.read_csv("val.csv", chunksize = 100))

cv = CountVectorizer(analyzer='char', ngram_range=(1, 5), dtype=np.float32, max_features=10000)
X_train, X_test, X_val = cv.fit_transform(train_data.text.tolist()), cv.transform(test_data.text.tolist()), cv.transform(val_data.text.tolist())  
Y_train, Y_test, Y_val = train_data.author.tolist(), test_data.author.tolist(), val_data.author.tolist()

# Convert labelled data into numbers
Encoder = LabelEncoder()
Y_train = Encoder.fit_transform(Y_train)
Y_test = Encoder.transform(Y_test)
Y_val = Encoder.transform(Y_val)

# **Data Normalization using MinMax Scaler**
X_train, X_test, X_val = X_train.toarray(), X_test.toarray(), X_val.toarray()

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

print(X_train_scaled)

# SVM without SDAE

model_names = ['Raw SVM', '1-layer SDAE SVM', '2-layers SDAE SVM', '3-layers SDAE SVM']
predicted_arr = []

svm = SVC(kernel='linear', C=1)
svm.fit(X_train_scaled, Y_train)
preds = svm.predict(X_test_scaled)

print("SVM Accuracy Score on test -> ", metrics.accuracy_score(preds, Y_test)*100)
print("SVM Accuracy Score on training -> ", metrics.accuracy_score(svm.predict(X_train_scaled), Y_train)*100)
print("SVM Accuracy Score on validation -> ", metrics.accuracy_score(svm.predict(X_val_scaled), Y_val)*100)

predicted_arr.append(preds)

# %% [markdown]
# SVM with SDAE
# ## 5. Pretraining and Finetuning

# %% [markdown]
# **Pretrain the denoising autoencoder**

# %%
# The "stacked" auto encoder will only contain 1 denoising auto encoder that will 
# transform the original input into 1000 units. Noise corruption is 0.3 and it uses 
# sigmoid activation for both encoder and decoder
stacked_auto_encoder_1 = StackedDenoisingAutoEncoder([1000], 0.3, 'sigmoid', 'sigmoid')
stacked_auto_encoder_1.pretrain(X_train_scaled, X_val_scaled, 20, 1)

# %%
# The "stacked" auto encoder will have 2 DAE's stacked, each with 1000 units.
stacked_auto_encoder_2 = StackedDenoisingAutoEncoder([1000, 1000], 0.3, 'sigmoid', 'sigmoid')
stacked_auto_encoder_2.pretrain(X_train_scaled, X_val_scaled, 20, 1)

# %%
# The "stacked" auto encoder will have 3 DAE's stacked, each with 1000 units.
stacked_auto_encoder_3 = StackedDenoisingAutoEncoder([1000, 1000, 1000], 0.3, 'sigmoid', 'sigmoid')
stacked_auto_encoder_3.pretrain(X_train_scaled, X_val_scaled, 20, 1)

# %% [markdown]
# ### SDAE Pretraining Plot

# %%
def plot_loss(history):
    # summarize history for loss
    # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

# %%
plot_loss(stacked_auto_encoder_1.history.history)

# %%
plot_loss(stacked_auto_encoder_2.history.history)

# %%
plot_loss(stacked_auto_encoder_3.history.history)

# %% [markdown]
# **Finetune the model by adding logistic regression layer**

# %%
# Create a 1 hot encoded Y_train
def create_one_hot(arr):
    out_arr = []
    for item in arr:
        max_arr = max(arr)
        one_hot_encoded = [0] * (max_arr+1)
        one_hot_encoded[item] = 1
        out_arr.append(one_hot_encoded)
    return np.array(out_arr)

Y_train_hot_encoded = create_one_hot(Y_train)
Y_val_hot_encoded = create_one_hot(Y_val)

# %%
fine_tuned_model_1, fine_tuned_encoder_1 = stacked_auto_encoder_1.finetune(X_train_scaled, Y_train_hot_encoded, X_val_scaled, Y_val_hot_encoded, 30, 1) 

# %%
fine_tuned_model_2, fine_tuned_encoder_2 = stacked_auto_encoder_2.finetune(X_train_scaled, Y_train_hot_encoded, X_val_scaled, Y_val_hot_encoded, 30, 1)

# %%
fine_tuned_model_3, fine_tuned_encoder_3 = stacked_auto_encoder_3.finetune(X_train_scaled, Y_train_hot_encoded, X_val_scaled, Y_val_hot_encoded, 30, 1) 

# %% [markdown]
# ### Plot of Finetuning Training

# %%
plot_loss(stacked_auto_encoder_1.history.history)

# %%
plot_loss(stacked_auto_encoder_2.history.history)

# %%
plot_loss(stacked_auto_encoder_3.history.history)

# %% [markdown]
# **Now feed the encoded representation into linear SVM**

# %%
fine_tuned_encoders = [fine_tuned_encoder_1, fine_tuned_encoder_2, fine_tuned_encoder_3]
for fine_tuned_encoder, model_name in zip(fine_tuned_encoders, model_names[1:]):
    print(model_name)
    X_train_encoded = fine_tuned_encoder.predict(X_train_scaled)

    # Fit to encoded data
    svm_autoencoder = SVC(kernel='linear', C=1, probability=True)
    svm_autoencoder.fit(X_train_encoded, Y_train)

    # Encode the test data and use SVM to predict its labels
    X_test_encoded = fine_tuned_encoder.predict(X_test_scaled)
    predicted = svm_autoencoder.predict(X_test_encoded)

    print("SVM Accuracy Score -> ", metrics.accuracy_score(predicted, Y_test)*100)
    predicted_arr.append(predicted)
    
    predicted_train = svm_autoencoder.predict(X_train_encoded)
    print("SVM Accuracy Score on training -> ", metrics.accuracy_score(predicted_train, Y_train)*100)

    X_val_encoded = fine_tuned_encoder.predict(X_val_scaled)
    predicted_val = svm_autoencoder.predict(X_val_encoded)
    print("SVM Accuracy Score on validation -> ", metrics.accuracy_score(predicted_val, Y_val)*100)

# %% [markdown]
# ## 6. Testing metrics

# %%
def testing_metrics(y_true, y_pred_arr, model_names, export=True):
    stats = {}
    for y_pred, model_name in zip(y_pred_arr, model_names):
        scores = metrics.classification_report(y_true, y_pred, zero_division=1, output_dict=True)

        stats[model_name] = {
            'cohen_kappa': metrics.cohen_kappa_score(y_true, y_pred),
            'matthews_corrcoef': metrics.matthews_corrcoef(y_true, y_pred),
            'micro-accuracy': scores['accuracy']
        }

        for metr1 in ('macro avg', 'weighted avg'):
            for metr2 in ('precision', 'recall', 'f1-score'):
                stats[model_name][metr1 + ' ' + metr2] = scores[metr1][metr2]
    
    stats_df = pd.DataFrame.from_dict(stats).T
    if export:
        stats_df.to_csv('svm_stats.csv')
    
    return stats_df

# %%
testing_metrics(Y_test, predicted_arr, model_names)


