from model.IA.DQN.dqn_agent import DQNAgent
from model.IA.Dataprocessing.dataProcess import process_realtime_data
from model.IA.environnement.TradEnvironnement import TradingEnv
import matplotlib.pyplot as plt


def plot_performance(profits_history, rewards_history):
    """Affiche les performances de l'entraînement."""
    plt.figure(figsize=(12, 6))

    # Profits cumulés
    plt.subplot(1, 2, 1)
    plt.plot(profits_history, label="Profits")
    plt.title("Profits cumulés")
    plt.xlabel("Episodes")
    plt.ylabel("Profits")
    plt.legend()

    # Récompenses moyennes
    plt.subplot(1, 2, 2)
    plt.plot(rewards_history, label="Récompenses moyennes")
    plt.title("Récompenses moyennes par épisode")
    plt.xlabel("Episodes")
    plt.ylabel("Récompense moyenne")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Liste des symboles à traiter
    symbols = ["BTCUSD"]

    # Traitement des données en temps réel
    realtime_data = process_realtime_data(symbols)
    print("Données récupérées :", realtime_data.keys())

    # Sélection des données du marché pour un symbole spécifique
    market_data = realtime_data["BTCUSD"]
    print(market_data)
    if market_data.empty:
        raise ValueError("Les données de marché sont vides après traitement. Vérifiez la préparation des données.")

    # Définition des dimensions de l'espace d'état et des actions
    state_size = market_data.shape[1] - 1# Exclure la colonne "time"
    print(state_size)
    action_size = 3  # Actions : Rester inactif, Acheter, Vendre

    # Initialisation de l'environnement de trading
    env = TradingEnv(market_data)

    # Initialisation de l'agent DQN
    agent = DQNAgent(state_size, action_size)

    # Variables pour le suivi des performances
    episodes = 100000  # Augmenter le nombre d'épisodes
    profits_history = []
    rewards_history = []

    # Entraînement de l'agent
    for episode in range(episodes):
        state = env.reset()
        total_rewards = 0

        # Simulation de l'épisode
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_rewards += reward

            if done:
                profits_history.append(info.get("profits", 0))
                rewards_history.append(total_rewards)
                print(f"Episode {episode + 1}/{episodes} - Total Rewards: {total_rewards:.2f}, Profits: {info.get('profits', 0):.2f}")
                break

        # Replay et mise à jour des paramètres
        agent.replay()
        # Mise à jour du modèle cible
        if episode % agent.update_target_every == 0:
            agent.update_target_model()

    plot_performance(profits_history, rewards_history)

    avg_reward = sum(rewards_history) / len(rewards_history) if len(rewards_history) > 0 else 0.0
    agent.save("trading_model.pth", episodes, avg_reward)

