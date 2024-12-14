from model.IA.Dataprocessing.dataProcess import prepareData
from model.IA.Analyzer.MarketAnalyzer import MarketAnalyzer

if __name__ == '__main__':
    symbols = ["EURUSD", "GBPUSD"]
    analyzer = MarketAnalyzer(symbols)

    try:
        analyzer.connect()
        market_data = analyzer.analyzeMarket()

        for symbol, df in market_data.items():
            print(f"\nDonnées pour {symbols} : \n", df.head(50))  # Augmentez le nombre de lignes
            processed_data = prepareData(df)
            print(processed_data.head())

    finally:
        analyzer.disconnect()
