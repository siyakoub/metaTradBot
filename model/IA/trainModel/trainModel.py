import os, sys
# Ajoute la racine du projet pour permettre les imports du package "model"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from model.IA.DQN.dqn_agent import DQNAgent
from model.IA.Dataprocessing.dataProcess import process_realtime_data
from model.IA.environnement.TradEnvironnement import TradingEnv
import matplotlib.pyplot as plt

def plot_performance(rewards_history):
    plt.figure(figsize=(10,4))
    plt.plot(rewards_history, label="Récompenses")
    plt.title("Récompense moyenne par épisode")
    plt.xlabel("Episodes")
    plt.ylabel("Récompense")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    symbols = ["BTCUSD"]
    realtime_data = process_realtime_data(symbols)
    market_data = realtime_data["BTCUSD"]
    if market_data.empty:
        raise ValueError("Données de marché vides")

    state_size = market_data.shape[1] - 1
    action_size = 3

    env = TradingEnv(market_data)
    agent = DQNAgent(state_size, action_size)

    # Appel au train() intégré
    episodes = 1000
    rewards_history = agent.train(env, episodes)

    # Trace des performances
    plot_performance(rewards_history)

    # Sauvegarde finale
    avg_reward = sum(rewards_history) / len(rewards_history)
    agent.save("trading_model_final.pth", episodes, avg_reward)
