from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd
from colorama import Fore


class MarketAnalyzer:

    def __init__(self, symbols):
        """
            Initialise le MarketAnalyzer avec une liste de symboles à analyser.

            :param symbols: Liste des symboles à surveiller (ex: ["EURUSD", "GBPUSD"])
        """
        self.symbols = symbols
        self.connected = False

    def connect(self):
        """
            Connecte l'instance au terminal MetaTrader 5.
        """
        if not mt5.initialize():
            raise ConnectionError(Fore.RED + f"La connexion avec MetaTrader 5 à échoué suite à l'erreur : {mt5.last_error()}")
        print(Fore.GREEN + "Connexion à MetaTrader 5 Réussi !")
        self.connected = True

    def disconnect(self):
        """
        Déconnecte l'instance de MetaTrader 5.
        :return:
        """
        mt5.shutdown()
        print(Fore.GREEN + "MetaTrader 5 à été déconnecté avec succès !")
        self.connected = False

    def get_market_data(self, symbol, timeframe=mt5.TIMEFRAME_H1, count=100):
        """
        Récupère les données de marché pour un symbole donné.
        :param symbol: Le symbole à récupérer (ex: "EURUSD").
        :param timeframe: Le timeframe des données (par défaut 1H).
        :param count: Le nombre de bougies à récupérer.
        :return: Un DataFrame contenant les données OHLC.
        """
        if not self.connected:
            raise ConnectionError(Fore.RED + "Non connecté à MetaTrader 5. Veuillez appeler connect() d'abord.")
        now = datetime.now()
        print(Fore.MAGENTA + f"Heure actuelle du système : {now}")
        # Ajouter un léger décalage si nécessaire pour éviter des données manquantes
        adjusted_time = now - timedelta(minutes=10)
        print(Fore.MAGENTA + f"Récupération des données jusqu'à : {adjusted_time}")
        # Obtenir les données récentes
        rates = mt5.copy_rates_from(symbol, timeframe, adjusted_time, count)
        if rates is None:
            raise ValueError(Fore.RED + f"Impossible de récupérer les données pour {symbol} : {mt5.last_error()}")
        # Convertir les données en DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def analyzeMarket(self):
        """
        Analyse les marchés pour tous les symboles configurés.
        :return: Un dictionnaire contenant les DataFrames des données de marché pour chaque symbole.
        """
        data = {}
        for symbol in self.symbols:
            try:
                print(Fore.YELLOW + f"Récupération des données pour {symbol}...")
                data[symbol] = self.get_market_data(symbol)
                print(Fore.LIGHTCYAN_EX + f"Données récupérées pour {symbol} avec succès.")
            except Exception as e:
                print(Fore.RED + f"Erreur lors de la récupération des données pour {symbol} : {e}")
        return data