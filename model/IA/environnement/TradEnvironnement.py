import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnv(gym.Env):
    def __init__(self, market_data, initial_balance=1000, window_size=20):
        """
        Environnement de trading avec observations séquentielles.

        :param market_data: Données de marché sous forme de DataFrame.
        :param initial_balance: Balance initiale pour le compte de trading.
        :param window_size: Taille de la fenêtre pour les observations séquentielles.
        """
        super(TradingEnv, self).__init__()
        self.market_data = market_data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.n_steps = len(market_data)

        # Actions possibles : 0 = rester inactif, 1 = acheter, 2 = vendre
        self.action_space = spaces.Discrete(3)

        # Observation : fenêtre glissante des prix et indicateurs techniques
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(window_size, market_data.shape[1] - 1),  # Exclure la colonne 'time'
            dtype=np.float32
        )

        # Variables internes
        self.balance = None
        self.current_step = None
        self.positions = None
        self.profits = None
        self.episode_rewards = None

    def reset(self):
        """
        Réinitialise l'environnement pour un nouvel épisode.
        """
        self.balance = self.initial_balance
        self.current_step = self.window_size  # Commence après les premières observations
        self.positions = []
        self.profits = 0
        self.episode_rewards = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.market_data.iloc[
              self.current_step - self.window_size: self.current_step
              ].drop("time", axis=1).values
        obs_min = obs.min(axis=0)
        obs_max = obs.max(axis=0)
        obs = (obs - obs_min) / (obs_max - obs_min + 1e-8)  # Normalisation
        obs = 2 * (obs - 0.5)  # Mise à l'échelle entre -1 et 1

        return obs.astype(np.float32)

    def step(self, action):
        """
        Effectue une action et met à jour l'état.
        """
        # Obtenez les données actuelles
        current_price = self.market_data.iloc[self.current_step]["close"]

        # Initialiser la récompense
        reward = 0

        # Exécuter l'action
        if action == 1:  # Acheter
            self.positions.append(current_price)
            reward = 0.05  # Récompense plus modérée pour l'achat (au lieu de pénalité)
        elif action == 2:  # Vendre
            if self.positions:  # Si une position est ouverte
                buy_price = self.positions.pop(0)
                profit = current_price - buy_price
                profit_percentage = (profit / buy_price) * 100 if buy_price != 0 else 0
                self.profits += profit
                reward = profit_percentage  # Récompense proportionnelle au pourcentage de profit
                self.balance += profit
            else:
                reward = -0.5  # Pénalité moins sévère pour vendre sans position ouverte
        else:  # Rester inactif
            reward = -0.005  # Pénalité plus légère pour rester inactif

        # Mise à jour de l'étape actuelle
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Observation suivante
        obs = self._next_observation()

        # Bonus en fin d'épisode
        if done:
            reward += self.profits  # Ajouter les profits totaux comme récompense finale

        # Accumuler les récompenses pour le suivi
        self.episode_rewards += reward

        # Informations supplémentaires pour le suivi
        info = {
            "profits": self.profits,
            "balance": self.balance,
            "positions": len(self.positions),
            "total_rewards": self.episode_rewards
        }

        return obs, reward, done, info

    def render(self, mode="human"):
        """
        Affiche l'état actuel de l'environnement.
        """
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, Profits: {self.profits:.2f}, Positions: {len(self.positions)}")
