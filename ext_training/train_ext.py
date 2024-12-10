import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
#from tensorflow.keras import initializers
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPool1D
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences 
from tf_keras import initializers
from tf_keras.models import Sequential
from tf_keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPool1D
import tf_keras

# Set the environment variable to use legacy Keras (Keras 2.x)
# os.environ['TF_USE_LEGACY_KERAS'] = '1'

print('tf_keras.__version__', tf_keras.__version__)

# Chargement des données
df = pd.read_csv('ext_training/train.csv')

# Création de la colonne cible 'toxic' (0 ou 1) en combinant toutes les colonnes de toxicité
df['toxic_label'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)

# Séparation des données en features (comment_text) et labels (toxic_label)
texts = df['comment_text'].fillna("").values
labels = df['toxic_label'].values
# Prétraitement des données textuelles
max_words = 20000  # Nombre maximum de mots à garder
max_len = 200  # Longueur maximale de chaque séquence de mots

# Tokenisation des textes
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Remplissage des séquences pour qu'elles aient toutes la même longueur
X = pad_sequences(sequences, maxlen=max_len)
y = labels

# Séparation des données en ensembles d'entraînement et de validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle
# TODO : Relancer ce script pour tester si le modèle est bien (j'ai mis input_length à un car sinon Error: Error when checking : expected embedding_input to have shape [null,200] but got array with shape [3,1].)
# TODO : Ne marche pas
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=1),
    LSTM(64, return_sequences=True, kernel_initializer=initializers.GlorotUniform(), recurrent_initializer=initializers.GlorotUniform()),
    GlobalMaxPool1D(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
batch_size = 32
epochs = 5
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

# Évaluation du modèle
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Accuracy: {accuracy * 100:.2f}%")

model.save('toxic_comment_model.h5')
tfjs.converters.save_keras_model(model, 'model_tfjs')
print("Modèle sauvegardé sous le nom 'toxic_comment_model.h5'")

# Prédiction d'un commentaire
def predict_toxicity(comment):
    sequence = tokenizer.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    return "Toxic" if prediction[0] > 0.5 else "Not Toxic"

# Exemple d'utilisation
print(predict_toxicity("FUCK YOU SON OF A BITCH I HATE YOU GO KILL YOURSELF PAR PITIE "))
