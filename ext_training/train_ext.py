import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPool1D

print("ça marche")

# Chargement des données
df = pd.read_csv('./train.csv')

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
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(64, return_sequences=True),
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
print("Modèle sauvegardé sous le nom 'toxic_comment_model.h5'")

# Prédiction d'un commentaire
def predict_toxicity(comment):
    sequence = tokenizer.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    return "Toxic" if prediction[0] > 0.5 else "Not Toxic"

# Exemple d'utilisation
print(predict_toxicity("FUCK YOU SON OF A BITCH I HATE YOU GO KILL YOURSELF PAR PITIE "))
