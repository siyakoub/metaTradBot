class RiskManager:
    def __init__(self, account_balance, risk_per_trade=1.0):
        """
        Initialise le gestionnaire de risque.
        :param account_balance: Solde total du compte de trading.
        :param risk_per_trade: Pourcentage de risque par trade (e.g., 1%).
        """
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade / 100  # Convertir en fraction (e.g., 1% devient 0.01)

    def calculate_position_size(self, stop_loss_distance, symbol_info):
        """
        Calcule la taille de la position basée sur le risque par trade et la distance SL.
        :param stop_loss_distance: Distance entre le prix d'entrée et le Stop Loss.
        :param symbol_info: Informations sur le symbole, incluant la valeur du pip.
        :return: Taille de la position en lots.
        """
        if stop_loss_distance <= 0:
            raise ValueError("La distance de Stop Loss doit être supérieure à zéro.")

        # Calcul du montant à risque
        risk_amount = self.account_balance * self.risk_per_trade

        # Valeur d'un pip pour la taille minimale de lot
        pip_value = symbol_info.point * symbol_info.trade_tick_value

        # Taille de position
        position_size = risk_amount / (stop_loss_distance * pip_value)

        # Normalisation pour respecter la taille minimale et les pas de volume
        position_size = max(symbol_info.volume_min, position_size)
        position_size = round(position_size / symbol_info.volume_step) * symbol_info.volume_step

        # Limite de taille maximale
        position_size = min(position_size, symbol_info.volume_max)

        return position_size

    def validate_stop_levels(self, entry_price, stop_loss, take_profit, symbol_info):
        """
        Valide que les niveaux de SL et TP respectent les exigences minimales.
        :param entry_price: Prix d'entrée de la position.
        :param stop_loss: Niveau de Stop Loss.
        :param take_profit: Niveau de Take Profit.
        :param symbol_info: Informations sur le symbole.
        :return: True si valide, sinon une exception est levée.
        """
        min_distance = symbol_info.trade_stops_level * symbol_info.point

        # Validation de la distance Stop Loss
        if stop_loss and abs(entry_price - stop_loss) < min_distance:
            raise ValueError(f"Le Stop Loss est trop proche du prix d'entrée. Distance minimale requise : {min_distance}")

        # Validation de la distance Take Profit
        if take_profit and abs(entry_price - take_profit) < min_distance:
            raise ValueError(f"Le Take Profit est trop proche du prix d'entrée. Distance minimale requise : {min_distance}")

        return True

    def adjust_risk_per_trade(self, new_risk_percentage):
        """
        Ajuste le pourcentage de risque par trade.
        :param new_risk_percentage: Nouveau pourcentage de risque (e.g., 2%).
        """
        if not (0 < new_risk_percentage <= 100):
            raise ValueError("Le pourcentage de risque doit être compris entre 0 et 100.")
        self.risk_per_trade = new_risk_percentage / 100
        print(f"Risque par trade ajusté à {new_risk_percentage}%.")

    def update_account_balance(self, new_balance):
        """
        Met à jour le solde du compte.
        :param new_balance: Nouveau solde total.
        """
        if new_balance <= 0:
            raise ValueError("Le solde du compte doit être supérieur à zéro.")
        self.account_balance = new_balance
        print(f"Solde du compte mis à jour : {new_balance}.")

