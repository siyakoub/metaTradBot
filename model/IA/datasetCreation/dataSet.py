import numpy as np


def create_dataset(data, lookback=30):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data.iloc[i:i + lookback].values)
        y.append(data.iloc[i + lookback]['target'])  # Exemple : 'target' est votre colonne cible
    return np.array(X), np.array(y)
