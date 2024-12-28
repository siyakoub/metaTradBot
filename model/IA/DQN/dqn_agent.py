import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):  # Reduced std_init for more stable exploration
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, input):
        if not self.training:
            return nn.functional.linear(input, self.weight_mu, self.bias_mu)
        weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
        bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        return nn.functional.linear(input, weight, bias)


class DuelingDQNetwork(nn.Module):
    def __init__(self, state_size, action_size, dropout_rate=0.2):
        super(DuelingDQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024)
        self.batch_norm1 = nn.LayerNorm(1024)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(1024, 512)
        self.batch_norm2 = nn.LayerNorm(512)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Separate value and advantage streams
        self.noisy_value1 = NoisyLinear(512, 256)
        self.value_bn = nn.LayerNorm(256)
        self.noisy_value2 = NoisyLinear(256, 1)

        self.noisy_advantage1 = NoisyLinear(512, 256)
        self.advantage_bn = nn.LayerNorm(256)
        self.noisy_advantage2 = NoisyLinear(256, action_size)  # Ensure it matches action_size

    def forward(self, x):
        x = self.dropout1(torch.relu(self.batch_norm1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.batch_norm2(self.fc2(x))))

        value = torch.relu(self.value_bn(self.noisy_value1(x)))
        value = self.noisy_value2(value)  # Output shape [batch_size, 1]

        advantage = torch.relu(self.advantage_bn(self.noisy_advantage1(x)))
        advantage = self.noisy_advantage2(advantage)  # Output shape [batch_size, action_size]

        return value + (advantage - advantage.mean(dim=1, keepdim=True))  # Output shape [batch_size, action_size]



Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_initial=0.4, beta_decay=0.0001):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # Increased importance sampling
        self.beta = beta_initial
        self.beta_decay = beta_decay
        self.max_priority = 1.0

    def push(self, *args):
        self.max_priority = max(self.max_priority, 1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == 0:
            return [], []  # Return empty lists for batch_size > 0

        priorities = self.priorities[:len(self.memory)]
        priorities = np.nan_to_num(priorities, nan=self.max_priority)

        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        priorities = np.ravel(priorities)  # Aflatir pour garantir un tableau 1D
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def update_beta(self):
        self.beta = min(1.0, self.beta + self.beta_decay)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparamètres optimisés
        self.memory = PrioritizedReplayBuffer(100000)  # Taille de buffer augmentée
        self.batch_size = 128  # Taille du batch augmentée
        self.gamma = 0.99  # Facteur de discount plus élevé
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Exploration plus élevée
        self.epsilon_decay = 0.995  # Décroissance plus lente
        self.learning_rate = 3e-4  # Taux d'apprentissage optimisé
        self.update_target_every = 1000  # Mises à jour des cibles plus fréquentes
        self.tau = 0.001  # Mises à jour douces des cibles

        # Double DQN
        self.model = DuelingDQNetwork(state_size, action_size).to(self.device)
        self.target_model = DuelingDQNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Planificateur de taux d'apprentissage
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        self.update_target_model()

    def update_target_model(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        self.model.eval()
        with torch.no_grad():
            # Assurez-vous que state est sous forme de tenseur et sur le bon appareil
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Échantillon avec priorité
        minibatch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Conversion en tenseurs
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        actions_tensor = torch.LongTensor(actions).view(-1, 1).to(
            self.device)
        dones_tensor = torch.FloatTensor(dones).view(-1, 1).to(self.device)
        weights_tensor = torch.FloatTensor(weights).view(-1, 1).to(self.device)

        # Calcul du target Double DQN
        with torch.no_grad():
            next_actions = self.model(next_states_tensor).argmax(1).unsqueeze(1)  # Shape [batch_size, 1]
            next_q_values = self.target_model(next_states_tensor).gather(1, next_actions)  # Shape [batch_size, 1]
            targets = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values  # Shape [batch_size, 1]

        # Q-values actuels
        current_q_values = self.model(states_tensor)  # Shape [batch_size, num_actions]

        # Si le modèle génère plusieurs valeurs (par exemple, dueling DQN), extraire la bonne valeur
        if current_q_values.dim() == 3:
            # Combinez les valeurs de l'état et les avantages pour obtenir une seule valeur Q par action
            current_q_values = current_q_values[:, :, 0] + current_q_values[:, :,
                                                           1]  # Exemples d'ajout des deux valeurs
            current_q_values = current_q_values.view(-1, self.action_size)

        # Vérifiez si la forme est correcte maintenant
        assert current_q_values.dim() == 2, f"Expected current_q_values to be of shape [batch_size, num_actions], but got {current_q_values.shape}"

        # Utiliser gather pour collecter les Q-values correspondant aux actions
        current_q_values = current_q_values.gather(1, actions_tensor)  # Shape [batch_size, 1]

        # Calcul de la perte Huber pondérée
        td_errors = torch.abs(current_q_values - targets)
        loss = (td_errors * weights_tensor).mean()

        # Étape d'optimisation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Mise à jour des priorités
        priorities = (td_errors.detach().cpu().numpy() + 1e-6) ** 0.6
        self.memory.update_priorities(indices, priorities)  # Assuming indices is a numpy array

        # Mise à jour du taux d'exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Mise à jour de beta pour l'échantillonnage d'importance
        self.memory.update_beta()

        return loss.item()

    def train(self, env, episodes, warmup_episodes=100):
        best_reward = float('-inf')
        episode_rewards = []

        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.replay()

            episode_rewards.append(total_reward)

            # Ajuster le taux d'apprentissage en fonction des performances
            self.scheduler.step(total_reward)

            # Sauvegarde du modèle
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.model.state_dict(), 'dqn_best_model.pth')

            print(f"Episode {e}/{episodes}, Reward: {total_reward}, Epsilon: {self.epsilon:.2f}")

        return episode_rewards

