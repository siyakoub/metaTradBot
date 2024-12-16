import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random


# Réseau bruyant
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
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


# Dueling DQN avec Réseau bruyant
class DuelingDQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.noisy_value1 = NoisyLinear(128, 128)
        self.noisy_value2 = NoisyLinear(128, 1)
        self.noisy_advantage1 = NoisyLinear(128, 128)
        self.noisy_advantage2 = NoisyLinear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = torch.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
        advantage = torch.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        return value + (advantage - advantage.mean())


# Prioritized Replay Buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, *args):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, alpha=0.6):
        priorities = self.priorities[:len(self.memory)]
        probabilities = priorities ** alpha / np.sum(priorities ** alpha)
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[i] for i in indices]
        return samples, indices

    def update_priorities(self, batch_indices, batch_priorities):
        batch_priorities = np.array(batch_priorities, dtype=np.float32).flatten()
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)


# DQN Agent
class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 3e-4
        self.update_target_every = 1000

        self.model = DuelingDQNetwork(state_size, action_size)
        self.target_model = DuelingDQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state_tensor)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch, indices = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states_tensor = torch.FloatTensor(np.array(states))
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(np.array(dones, dtype=float))

        with torch.no_grad():
            next_actions = self.model(next_states_tensor).argmax(dim=1)
            next_q_values = self.target_model(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        q_values = self.model(states_tensor).gather(1, torch.tensor(actions).unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for idx, td_error in zip(indices, (loss.view(-1).detach().numpy() + 0.01)):
            self.memory.update_priorities(indices, td_error)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes):
        for e in range(episodes):
            state = env.reset()
            for time in range(env.n_steps):
                action = self.act(state)
                next_state, reward, done = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print(f"Episode: {e}/{episodes}, Score: {time}")
                    break
            self.replay()
            if e % self.update_target_every == 0:
                self.update_target_model()
