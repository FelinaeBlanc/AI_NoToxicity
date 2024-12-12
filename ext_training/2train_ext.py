import pandas as pd
import tensorflowjs as tfjs
import os
import numpy as np
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras import initializers
from tf_keras.models import Sequential
from tf_keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

# Charger les données
df = pd.read_csv('ext_training/train.csv')

# Créer une nouvelle colonne 'toxic_label' (0 ou 1) en combinant toutes les colonnes de toxicité
df['toxic_label'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)

# Séparer les données en features (comment_text) et labels (toxic_label)
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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculer les poids de classes pour traiter le déséquilibre des classes
class_weights = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Création du modèle
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=True, kernel_initializer=initializers.GlorotUniform(), recurrent_initializer=initializers.GlorotUniform()),
    GlobalMaxPool1D(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
batch_size = 32
epochs = 5
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), class_weight=class_weight_dict)

# Évaluation du modèle
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Prédictions sur le jeu de validation
y_pred = model.predict(X_val)
y_pred = (y_pred > 0.5).astype(int)  # Convertir les probabilités en labels binaires

# Afficher le rapport de classification
print("Classification Report (Precision, Recall, F1-Score):")
print(classification_report(y_val, y_pred))

# Sauvegarder le modèle en format TensorFlow.js
tfjs.converters.save_keras_model(model, 'model_test')

