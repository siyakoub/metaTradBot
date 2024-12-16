from model.IA.Dataprocessing.dataProcess import get_realtime_data

if __name__ == '__main__':
    # Liste des symboles à analyser
    symbols = ["EURUSD", "GBPUSD"]

    # Récupération et préparation des données en temps réel
    print("⏳ Récupération des données en temps réel...")
    realtime_data = get_realtime_data(symbols)

    # Traitement des données récupérées
    for symbol, df in realtime_data.items():
        print(f"\n=== Traitement des données pour {symbol} ===")
        if df.empty:
            print(f"⚠️ Données insuffisantes pour {symbol}.")
            continue

        # Vérification de la qualité des données
        print(f"🔍 Vérification des données pour {symbol}...")
        print(f"Résumé initial des données pour {symbol} :")
        print(df.info())  # Affiche un résumé des colonnes et des NaN
        print(df.describe())  # Données statistiques pour valider les valeurs

        # Affiche les dernières lignes pour une validation manuelle
        print(f"✅ Données prêtes pour {symbol} :")
        print(df.tail())

    print("\n=== Données finales pour tous les symboles ===")
    print(realtime_data)
