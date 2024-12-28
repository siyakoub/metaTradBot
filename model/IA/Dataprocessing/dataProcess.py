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
    print("Données brutes avant traitement :")
    print(data.head(20))  # Vérifiez les premières lignes

    # Vérifiez les colonnes avec des NaN
    print("Colonnes avec valeurs manquantes avant traitement :")
    print(data.isna().sum())

    # Normalisation
    scaler = MinMaxScaler()
    columns_to_normalize = ['open', 'high', 'low', 'close']
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

    # Ajout des indicateurs techniques
    data['ma_fast'] = data['close'].rolling(window=20).mean()
    data['ma_slow'] = data['close'].rolling(window=50).mean()
    data['rsi'] = calculate_rsi(data['close'], period=14)

    print("Données après calcul des indicateurs :")
    print(data.head(20))  # Vérifiez les données après ajout des indicateurs

    # Vérifiez les NaN après calcul des indicateurs
    print("Colonnes avec valeurs manquantes après ajout des indicateurs :")
    print(data.isna().sum())

    # Supprimez les valeurs NaN résultant des calculs d'indicateurs
    data.dropna(inplace=True)
    print("Données après suppression des NaN :")
    print(data.head(20))

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


def check_data_quality(data):
    print("Résumé des données après ajout des indicateurs :")
    print(data.info())  # Vérifie les colonnes et les NaN
    print(data.describe())  # Données statistiques pour valider les valeurs
    missing = data.isna().sum()
    print(f"Colonnes avec valeurs manquantes :\n{missing[missing > 0]}")  # Affiche les colonnes problématiques


def normalize_data(df):
    df_normalized = df.copy()
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):  # Applique uniquement aux colonnes numériques
            df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df_normalized



def process_realtime_data(symbols):
    # Récupération des données en temps réel
    realtime_data = get_realtime_data(symbols)

    # Initialiser un ExcelWriter pour écrire dans un fichier Excel
    with pd.ExcelWriter('donnees_realtime.xlsx') as writer:
        # Traitement des données
        for symbol, df in realtime_data.items():
            print(f"Traitement des données pour {symbol}...")

            if df.empty:
                print(f"⚠️ Données insuffisantes pour {symbol}")
                continue

            # Normalisation des données
            df = normalize_data(df)

            # Calcul des indicateurs
            df['ma_fast'] = df['close'].rolling(window=5).mean()
            df['ma_slow'] = df['close'].rolling(window=10).mean()
            df['rsi'] = calculate_rsi(df['close'], period=14)

            # Gestion des valeurs manquantes
            # Suppression des colonnes inutiles si elles sont complètement vides
            df = df.drop(columns=['real_volume'], errors='ignore')

            # Forward fill puis backward fill pour ma_fast et ma_slow
            df['ma_fast'] = df['ma_fast'].ffill().bfill()
            df['ma_slow'] = df['ma_slow'].ffill().bfill()

            # Remplissage de 'rsi' avec la moyenne
            rsi_mean = df['rsi'].mean()
            df['rsi'] = df['rsi'].fillna(rsi_mean)

            # Vérification finale des NaN
            print("Vérification des données après traitement :")
            print(df.isnull().sum())

            # Filtrer les lignes restantes
            df = df.dropna()
            print("Nombre de lignes restantes :", len(df))

            # Vérification de la qualité des données finales
            check_data_quality(df)

            print(f"✅ Données prêtes pour {symbol} :")
            print(df.tail())  # Affiche les dernières lignes

            # Mise à jour des données dans le dictionnaire
            realtime_data[symbol] = df

            # Enregistrer le DataFrame dans un fichier Excel, avec le symbole comme nom de feuille
            df.to_excel(writer, sheet_name=symbol)
            print("Données enregistré !")
            print(df)

    return realtime_data

