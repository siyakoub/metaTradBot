from model.IA.Dataprocessing.dataProcess import get_realtime_data

if __name__ == '__main__':
    # Exemple : récupérer les données en temps réel pour EURUSD et GBPUSD
    symbols = ["EURUSD", "GBPUSD"]
    realtime_data = get_realtime_data(symbols)
    print("Données récupérer en temps réel : ")
    print(realtime_data)
