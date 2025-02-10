import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
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

        self.fc_intermediate = nn.Linear(512, 512)
        self.batch_norm_intermediate = nn.LayerNorm(512)
        self.dropout_intermediate = nn.Dropout(dropout_rate)

        self.noisy_value1 = NoisyLinear(512, 256)
        self.value_bn = nn.LayerNorm(256)
        self.noisy_value2 = NoisyLinear(256, 1)

        self.noisy_advantage1 = NoisyLinear(512, 256)
        self.advantage_bn = nn.LayerNorm(256)
        self.noisy_advantage2 = NoisyLinear(256, action_size)

    def forward(self, x):
        x = self.dropout1(torch.relu(self.batch_norm1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.batch_norm2(self.fc2(x))))
        x = self.dropout_intermediate(torch.relu(self.batch_norm_intermediate(self.fc_intermediate(x))))

        value = torch.relu(self.value_bn(self.noisy_value1(x)))
        value = self.noisy_value2(value)

        advantage = torch.relu(self.advantage_bn(self.noisy_advantage1(x)))
        advantage = self.noisy_advantage2(advantage)

        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()


Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_initial=0.4, beta_decay=0.0001):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
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

    def add(self, experience, td_error):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        self.priorities[self.position] = max_priority
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
        priorities = np.ravel(priorities)
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
        self.memory = PrioritizedReplayBuffer(
            capacity=15000,  # Augmenté pour stocker plus d'expériences de trading
            alpha=0.7,  # Augmenté pour donner plus d'importance aux expériences prioritaires
            beta_initial=0.5,  # Augmenté pour un meilleur échantillonnage
            beta_decay=0.0005  # Ajusté pour atteindre beta=1 plus rapidement
        )
        # Hyperparamètres ajustés pour 50000 épisodes
        self.batch_size = 512  # Augmenté pour plus de stabilité
        self.gamma = 0.98  # Maintenu pour le trading 5min
        self.epsilon = 1.0
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.99999  # Très lent decay pour maintenir l'exploration plus longtemps
        self.learning_rate = 5e-5  # Réduit pour une convergence plus stable sur le long terme
        self.update_target_every = 200  # Augmenté pour plus de stabilité
        self.tau = 0.001  # Réduit pour des mises à jour plus douces
        self.model = DuelingDQNetwork(state_size, action_size).to(self.device)
        self.target_model = DuelingDQNetwork(state_size, action_size).to(self.device)
        # Optimizer avec paramètres pour apprentissage long
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,  # Régularisation ajustée
            betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=500,  # Augmenté significativement pour 50000 épisodes
            min_lr=1e-7,  # LR minimum plus bas
            threshold=1e-4  # Seuil de changement plus strict
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

        self.model.reset_noise()
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        self.model.reset_noise()
        self.target_model.reset_noise()

        minibatch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        actions_tensor = torch.LongTensor(actions).view(-1, 1).to(self.device)  # Shape (batch_size, 1)
        dones_tensor = torch.FloatTensor(dones).view(-1, 1).to(self.device)
        weights_tensor = torch.FloatTensor(weights).view(-1, 1).to(self.device)

        with torch.no_grad():
            next_actions = self.model(next_states_tensor).argmax(1).unsqueeze(1)
            next_q_values = self.target_model(next_states_tensor).gather(1, next_actions)
            targets = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

        current_q_values = self.model(states_tensor)

        print("current_q_values shape:", current_q_values.shape)
        print("actions_tensor shape:", actions_tensor.shape)

        # Assurez-vous que les actions sont dans les bonnes limites
        actions_tensor = actions_tensor.clamp(0, self.action_size - 1)

        # Ajustez les indices pour gather
        actions_tensor_expanded = actions_tensor.unsqueeze(2)  # Shape (batch_size, 1, 1)

        # Maintenant, vous pouvez utiliser gather correctement
        chosen_q_values = current_q_values.gather(1, actions_tensor_expanded)  # Shape (batch_size, 1, 1)

        # Compute the TD error
        td_errors = torch.abs(chosen_q_values.squeeze(1) - targets)  # Shape (batch_size)

        # Compute loss
        loss = (td_errors * weights_tensor).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        priorities = np.clip((td_errors.detach().cpu().numpy() + 1e-6) ** 0.6, a_min=1e-5, a_max=1.0)
        self.memory.update_priorities(indices, priorities)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.memory.update_beta()
        print(f"epsilon: {self.epsilon:.4f}, loss: {loss.item():.4f}")

        return loss.item()

    def train(self, env, episodes=50000):
        best_reward = float('-inf')
        episode_rewards = []

        # Paramètres d'entraînement adaptés pour 50000 épisodes
        evaluation_window = 500  # Fenêtre d'évaluation plus large
        early_stopping_patience = 2000  # Patience augmentée significativement
        min_experiences = 5000  # Plus d'expériences avant de commencer

        no_improvement_count = 0
        best_avg_reward = float('-inf')

        # Sauvegarde des meilleurs modèles
        best_model_reward = float('-inf')
        save_interval = 1000  # Sauvegarde tous les 1000 épisodes

        for e in range(episodes):
            self.model.reset_noise()
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)

                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(self.memory) > min_experiences:
                    self.replay()

            episode_rewards.append(total_reward)
            avg_reward = np.mean(episode_rewards[-evaluation_window:])

            # Mise à jour du scheduler
            self.scheduler.step(avg_reward)

            # Early stopping avec plus de patience
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Sauvegarde du meilleur modèle
            if avg_reward > best_model_reward:
                best_model_reward = avg_reward
                torch.save({
                    'episode': e,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'reward': avg_reward,
                }, f'best_model.pth')

            # Sauvegarde périodique
            if e % save_interval == 0:
                torch.save({
                    'episode': e,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'reward': avg_reward,
                }, f'checkpoint_episode_{e}.pth')

            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping at episode {e}")
                break

            if total_reward > best_reward:
                best_reward = total_reward
                self.update_target_model()

            # Affichage plus fréquent des métriques
            if e % 100 == 0:
                print(
                    f"Episode {e}/{episodes} - Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")

        return episode_rewards

    def save(self, filepath, episode, reward):
        torch.save({
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward': reward,
        }, filepath)

