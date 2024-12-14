import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model.IA.datasetCreation.dataSet import create_dataset


def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes : Vente, Neutre, Achat
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Charger et préparer les données
data = ...  # Vos données ici
X, y = create_dataset(data, lookback=30)

# Encodage des labels pour une classification multi-classes
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construction du modèle
model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

# Entraînement
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)

# Évaluation
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_test_classes = y_test.argmax(axis=1)
print(classification_report(y_test_classes, y_pred_classes))
model.save('trading_model.h5')
