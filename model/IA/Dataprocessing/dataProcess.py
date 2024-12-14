import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from model.IA.Analyzer.MarketAnalyzer import MarketAnalyzer


def calculate_rsi(prices, period=14):
    """
    Calcule l'indicateur RSI (Relative Strength Index).
    :param prices: Liste ou série des prix de clôture.
    :param period: Période pour le calcul du RSI (par défaut 14).
    :return: Série RSI.
    """
    prices = np.array(prices)
    if len(prices) < period:
        # Retourner une série de NaN si les données sont insuffisantes
        return np.full_like(prices, np.nan, dtype=np.float64)

    deltas = np.diff(prices)

    # Gains et pertes
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Moyenne mobile exponentielle
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    # Gestion explicite des cas où avg_loss est 0
    rs = np.zeros_like(avg_gain)
    rs[avg_loss != 0] = avg_gain[avg_loss != 0] / avg_loss[avg_loss != 0]
    rs[avg_loss == 0] = np.inf  # Si aucune perte, rs est infini

    rsi = np.zeros_like(rs)
    rsi[avg_loss != 0] = 100 - (100 / (1 + rs[avg_loss != 0]))
    rsi[avg_loss == 0] = 100  # RSI à 100 en cas de gains uniquement

    # Remplir les premières valeurs avec NaN
    rsi[:period] = np.nan

    return rsi




def prepareData(data):
    """
    Prépare les données pour l'analyse.
    :param data: DataFrame contenant les données brutes.
    :return: DataFrame avec les données normalisées et les indicateurs calculés.
    """
    data = data.copy()  # Crée une copie explicite pour éviter les warnings
    scaler = MinMaxScaler()
    data.loc[:, ['open', 'high', 'low', 'close']] = scaler.fit_transform(data[['open', 'high', 'low', 'close']])

    # Vérification de la taille des données
    if len(data) < 20:
        print(f"Pas assez de données pour calculer les indicateurs. Lignes disponibles : {len(data)}")
        return data

    # Ajout des indicateurs techniques
    data['ma_fast'] = data['close'].rolling(window=20).mean()
    data['ma_slow'] = data['close'].rolling(window=50).mean()
    data['rsi'] = calculate_rsi(data['close'], period=14)

    # Suppression des valeurs NaN générées par les indicateurs
    data.dropna(inplace=True)

    return data

def get_realtime_data(symbols):
    analyzer = MarketAnalyzer(symbols)
    try:
        analyzer.connect()
        market_data = analyzer.analyzeMarket()
        prepared_data = {}
        for symbol, df in market_data.items():
            prepared_data[symbol] = prepareData(df)  # Préparer les données avec vos fonctions existantes
        return prepared_data
    finally:
        analyzer.disconnect()


