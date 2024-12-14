import MetaTrader5 as mt5
from colorama import Fore

class TradeManager:
    def __init__(self):
        """
        Initialise l'objet TradeManager avec l'état de connexion de MetaTrader 5
        """
        self.connected = False

    def connect(self):
        """
        Connecte l'instance au terminal MetaTrader 5
        """
        if not mt5.initialize():
            raise ConnectionError(
                f"{Fore.RED}Erreur lors de l'initialisation de la connexion au terminal MetaTrader 5 : {mt5.last_error()}"
            )
        print(f"{Fore.GREEN}Connecté à MetaTrader 5")
        self.connected = True

    def disconnect(self):
        """
        Déconnecte l'instance du terminal MetaTrader 5
        """
        mt5.shutdown()
        print(f"{Fore.YELLOW}Déconnecté de MetaTrader 5")
        self.connected = False

    def openTrade(self, symbol, action, volume, stopLoss=None, takeProfit=None, deviation=10):
        """
        Ouvre une position sur le marché.
        :param symbol: Le symbole de l'instrument (e.g., "EURUSD").
        :param action: "buy" pour achat ou "sell" pour vente.
        :param volume: Volume de la position (e.g., 0.1).
        :param stopLoss: Niveau de Stop Loss (optionnel).
        :param takeProfit: Niveau de Take Profit (optionnel).
        :param deviation: Tolérance de prix en points.
        :return: Résultat de la requête.
        """
        if not self.connected:
            raise ConnectionError(f"{Fore.RED}Non connecté à MetaTrader 5 : {mt5.last_error()}")

        # Récupération des informations sur le symbole
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            raise ValueError(f"{Fore.RED}Symbole introuvable : {symbol}")

        # Activer le trading sur le symbole s'il n'est pas activé
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise ValueError(f"{Fore.RED}Impossible d'activer le trading pour le symbole : {symbol}")

        # Distance minimale pour SL/TP
        min_distance = symbol_info.trade_stops_level * symbol_info.point

        try:
            # Déterminer le prix actuel et le type d'ordre
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                raise ValueError(f"{Fore.RED}Impossible de récupérer les tick data pour : {symbol}")

            if action == 'buy':
                price = tick.ask
                action_type = mt5.ORDER_TYPE_BUY
            else:  # action == 'sell'
                price = tick.bid
                action_type = mt5.ORDER_TYPE_SELL

            # Ajustement des distances de Stop Loss et Take Profit si nécessaire
            if stopLoss is not None:
                if action == 'buy' and stopLoss > price - min_distance:
                    stopLoss = price - min_distance
                    print(f"{Fore.YELLOW}Stop Loss ajusté à : {stopLoss}")
                elif action == 'sell' and stopLoss < price + min_distance:
                    stopLoss = price + min_distance
                    print(f"{Fore.YELLOW}Stop Loss ajusté à : {stopLoss}")

            if takeProfit is not None:
                if action == 'buy' and takeProfit < price + min_distance:
                    takeProfit = price + min_distance
                    print(f"{Fore.YELLOW}Take Profit ajusté à : {takeProfit}")
                elif action == 'sell' and takeProfit > price - min_distance:
                    takeProfit = price - min_distance
                    print(f"{Fore.YELLOW}Take Profit ajusté à : {takeProfit}")

            broker = mt5.account_info().company
            print(broker)
            if broker == "STARTRADER International PTY Limited":
                filling_mode = mt5.ORDER_FILLING_IOC
            elif broker == "MetaQuotes Ltd.":
                filling_mode = mt5.ORDER_FILLING_RETURN
            else:
                filling_mode = mt5.ORDER_FILLING_FOK  # Mode par défaut

            # Construction de la requête
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": action_type,
                "price": price,
                "sl": stopLoss,
                "tp": takeProfit,
                "type_filling": filling_mode,
                "deviation": deviation,
                "magic": 234000,
                "comment": "Trade ouvert par MetaTradBot"
            }

            # Logs pour la requête
            print(f"Requête envoyée : {request}")

            # Envoi de la requête
            result = mt5.order_send(request)

            # Vérification si result est None
            if result is None:
                raise ValueError(f"{Fore.RED}Erreur lors de l'envoi de l'ordre: order_send() a renvoyé None")

            # Vérification si le marché est ouvert
            if not mt5.symbol_info(symbol).trade_mode:
                raise ValueError(f"{Fore.RED}Marché fermé pour {symbol}")

            # Vérification des résultats
            if result.retcode != mt5.TRADE_RETCODE_DONE or result.retcode is None:
                print(f"{Fore.RED}Echec de l'ouverture de la position : {result.retcode} - {result.comment}")
                # Logique pour gérer l'erreur (par exemple, réessayer avec des valeurs ajustées)
                raise ValueError(f"{Fore.RED}Echec de l'ouverture de la position : {result.retcode}")

            print(f"{Fore.GREEN}Trade exécuté avec succès : {result}")
            return result

        except ValueError as e:
            if "10018" in str(e):  # Vérifier si l'erreur est due au marché fermé
                print(f"{Fore.YELLOW}Impossible d'ouvrir la position : {e}")
                # Logique pour gérer le marché fermé (par exemple, réessayer plus tard)
                # ...
            else:
                print(f"{Fore.RED}Erreur lors de l'envoi de l'ordre: {e}")
                raise  # Relève l'exception pour les autres erreurs

        except Exception as e:
            print(f"{Fore.RED}Erreur lors de l'envoi de l'ordre: {e}")
            raise  # Relève l'exception pour arrêter l'exécution


    def closeTrade(self, ticket, deviation=10):
        """
        Ferme une position existante en utilisant son ticket.
        :param ticket: Identifiant unique de la position.
        :param deviation: Tolérance de prix en points.
        :return: Détails de la fermeture ou une exception.
        """
        if not self.connected:
            raise ConnectionError(f"{Fore.RED}Non connecté à MetaTrader 5 : {mt5.last_error()}")

        # Récupération des informations de la position
        position = mt5.positions_get(ticket=ticket)
        if not position:
            raise ValueError(f"{Fore.RED}Aucune position trouvée pour le ticket : {ticket}")

        position = position[0]
        symbol = position.symbol
        volume = position.volume

        # Déterminer le type d'ordre de clôture et le prix actuel
        if position.type == mt5.ORDER_TYPE_BUY:
            action_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid  # Prix Bid pour la vente
        else:  # position.type == mt5.ORDER_TYPE_SELL
            action_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask  # Prix Ask pour l'achat

        # Corps de la requête
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": action_type,
            "position": position.ticket,  # Ajout de l'ID de position
            "price": price,
            "deviation": deviation,
            "magic": 234000,
            "comment": "Trade fermé par MetaTradBot",
        }

        try:
            # Envoi de la requête
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"{Fore.RED}Echec de la fermeture de la position : {result.retcode} - {result.comment}")
                # Tentative de fermeture forcée en cas d'échec initial
                if result.retcode == 10016:  # Code d'erreur pour "Invalid stops"
                    print(f"{Fore.YELLOW}Tentative de fermeture forcée de la position...")
                    request["sl"] = 0  # Suppression du Stop Loss
                    request["tp"] = 0  # Suppression du Take Profit
                    result = mt5.order_send(request)
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"{Fore.RED}Echec de la fermeture forcée : {result.retcode} - {result.comment}")
                        raise ValueError(f"{Fore.RED}Echec de la fermeture de la position : {result.retcode}")
                    else:
                        print(f"{Fore.GREEN}Position fermée avec succès (forcée) : {result}")
                else:
                    raise ValueError(f"{Fore.RED}Echec de la fermeture de la position : {result.retcode}")

            print(f"{Fore.GREEN}Position fermée avec succès : {result}")
            return result
        except Exception as e:
            print(f"{Fore.RED}Erreur lors de l'envoi de l'ordre: {e}")
            raise  # Relève l'exception pour arrêter l'exécution