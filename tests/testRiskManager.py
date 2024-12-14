# Exemple d'utilisation
from model.IA.RiskManager.riskManager import RiskManager

if __name__ == "__main__":
    # Création d'une instance de RiskManager
    risk_manager = RiskManager(account_balance=10000, risk_per_trade=1)

    # Exemple d'informations sur un symbole
    class MockSymbolInfo:
        point = 0.0001
        trade_tick_value = 10  # Valeur d'un pip pour 1 lot
        volume_min = 0.01
        volume_max = 100
        volume_step = 0.01
        trade_stops_level = 10  # En pips

    symbol_info = MockSymbolInfo()

    # Calcul de la taille de position
    try:
        position_size = risk_manager.calculate_position_size(stop_loss_distance=0.001, symbol_info=symbol_info)
        print(f"Taille de la position : {position_size} lots")
    except ValueError as e:
        print(e)

    # Validation des niveaux de SL/TP
    try:
        risk_manager.validate_stop_levels(entry_price=1.2000, stop_loss=1.1980, take_profit=1.2050, symbol_info=symbol_info)
        print("Niveaux de SL et TP valides.")
    except ValueError as e:
        print(e)