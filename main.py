import time
import MetaTrader5 as mt5

if __name__ == '__main__':
    # Initialisation de MetaTrader 5
    if not mt5.initialize():
        print(f"Erreur d'initialisation : {mt5.last_error()}")
        quit()

    # Vérification des informations de la plateforme
    print("Informations sur MetaTrader 5 :")
    print(mt5.terminal_info())
    print(f"Version : {mt5.version()}")

    # Paramètres principaux
    symbol = "EURUSD"
    lot = 0.01  # Taille par défaut du lot
    price_usd = 40  # Taille de la position en USD
    deviation = 20  # Tolérance de slippage

    # Vérification si le symbole est disponible
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Erreur : Le symbole {symbol} n'est pas trouvé.")
        mt5.shutdown()
        quit()

    account_info = mt5.account_info()
    if account_info is None:
        print("Impossible de récupérer les informations du compte")
        mt5.shutdown()
        quit()

    # Activer le symbole s'il n'est pas visible
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"Erreur : Impossible d'activer {symbol}")
            mt5.shutdown()
            quit()

    # Récupération des informations actuelles du marché
    symbol_tick = mt5.symbol_info_tick(symbol)
    if symbol_tick is None or symbol_tick.ask <= 0 or symbol_tick.bid <= 0:
        print(f"Erreur : Impossible de récupérer les ticks pour {symbol}")
        mt5.shutdown()
        quit()

    print(account_info)

    # Calcul du volume en lots en fonction du montant USD
    volume = round(price_usd / symbol_tick.ask, 2)
    volume_step = symbol_info.volume_step
    # Limitation du volume si la marge est insuffisante
    volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))

    # Afficher le volume calculé
    print(f"Volume calculé : {volume} lots")

    # Préparation de la requête d'ouverture
    request_open = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": symbol_tick.ask,
        "sl": symbol_tick.ask - 100 * symbol_info.point,
        "tp": symbol_tick.ask + 100 * symbol_info.point,
        "deviation": deviation,
        "magic": 234000,
        "comment": "Script Trade Open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,  # Changer ici
    }

    # Envoi de la requête d'ouverture
    result_open = mt5.order_send(request_open)
    if result_open.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Erreur : Impossible d'ouvrir la position. Code : {result_open.retcode}")
        result_dict = result_open._asdict()
        for field, value in result_dict.items():
            print(f"   {field} : {value}")
        mt5.shutdown()
        quit()

    print(f"Position ouverte avec succès : {result_open}")

    # Attente de quelques secondes avant de fermer la position
    time.sleep(10)

    # Récupération de l'ID de la position ouverte
    position_id = result_open.order

    # Préparation de la requête de fermeture
    request_close = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL,
        "position": position_id,
        "price": symbol_tick.bid,
        "deviation": deviation,
        "magic": 234000,
        "comment": "Script Trade Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,  # Changer ici
    }

    # Envoi de la requête de fermeture
    result_close = mt5.order_send(request_close)
    if result_close.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Erreur : Impossible de fermer la position. Code : {result_close.retcode}")
        result_dict = result_close._asdict()
        for field, value in result_dict.items():
            print(f"   {field} : {value}")
    else:
        print(f"Position fermée avec succès : {result_close}")

    # Déconnexion de MetaTrader 5
    mt5.shutdown()
