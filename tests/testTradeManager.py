import time
import MetaTrader5 as mt5

from model.IA.TradeManager.TradeManager import TradeManager

if __name__ == "__main__":
    tm = TradeManager()

    try:
        tm.connect()

        # Exemple : ouvrir un trade
        result = tm.openTrade("EURUSD", "buy", 0.1, stopLoss=1.05, takeProfit=1.10)

        # Vérifier que le trade s'est bien ouvert
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print("Trade ouvert avec succès. Attente de 30 secondes avant fermeture...")
            time.sleep(30)

            # Fermer le trade
            ticket = result.order  # Récupération du ticket du trade ouvert
            close_result = tm.closeTrade(ticket)
            print(f"Résultat de la fermeture : {close_result}")
        else:
            print("Impossible d'ouvrir le trade.")

    finally:
        tm.disconnect()