from model.IA.Analyzer.MarketAnalyzer import MarketAnalyzer

if __name__ == '__main__':
    symbols = ["EURUSD", "GBPUSD"]

    analyzer = MarketAnalyzer(symbols)

    try:
        analyzer.connect()
        market_data = analyzer.analyzeMarket()

        for symbol, df in market_data.items():
            print(f"\nDonnées pour {symbols} : \n", df.head())

    finally:
        analyzer.disconnect()