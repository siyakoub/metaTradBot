import unittest
import torch
import numpy as np
from model.IA.DQN.dqn_agent import DuelingDQNetwork, PrioritizedReplayBuffer, DQNAgent


class TestDQNModel(unittest.TestCase):
    def setUp(self):
        # Configurations de test
        self.state_size = 8
        self.action_size = 4
        self.batch_size = 64
        self.memory_capacity = 10000

        # Initialisation du réseau et de l'agent
        self.model = DuelingDQNetwork(self.state_size, self.action_size)
        self.memory = PrioritizedReplayBuffer(self.memory_capacity)
        self.agent = DQNAgent(self.state_size, self.action_size)

    def test_noisy_layer_forward(self):
        """Test de passage avant (forward) du réseau avec Noisy Layers"""
        state = torch.rand(1, self.state_size)  # État aléatoire
        output = self.model(state)
        self.assertEqual(output.shape, (1, self.action_size),
                         "La sortie du modèle ne correspond pas à la taille des actions")

    def test_memory_push_and_sample(self):
        """Test d'ajout et d'échantillonnage dans le buffer"""
        state = np.random.random(self.state_size)
        action = 1
        reward = 1.0
        next_state = np.random.random(self.state_size)
        done = False

        # Ajout d'une expérience
        self.memory.push(state, action, reward, next_state, done)
        self.assertEqual(len(self.memory.memory), 1, "La mémoire ne contient pas l'expérience ajoutée")

        # Échantillonnage
        samples, indices = self.memory.sample(self.batch_size)
        self.assertGreater(len(samples), 0, "Aucun échantillon n'a été retourné")
        self.assertEqual(len(indices), len(samples), "Les indices et les échantillons ne correspondent pas")

    def test_agent_act(self):
        """Test de la méthode act()"""
        state = np.random.random(self.state_size)
        action = self.agent.act(state)
        self.assertTrue(0 <= action < self.action_size, "L'action retournée n'est pas valide")

    def test_agent_remember(self):
        """Test de la méthode remember()"""
        state = np.random.random(self.state_size)
        action = 2
        reward = 1.0
        next_state = np.random.random(self.state_size)
        done = False

        self.agent.remember(state, action, reward, next_state, done)
        self.assertGreater(len(self.agent.memory.memory), 0, "La mémoire de l'agent ne contient pas d'expérience")

    def test_agent_replay(self):
        """Test de la méthode replay()"""
        # Remplir la mémoire
        for _ in range(self.batch_size):
            state = np.random.random(self.state_size)
            action = np.random.randint(self.action_size)
            reward = np.random.random()
            next_state = np.random.random(self.state_size)
            done = np.random.choice([True, False])
            self.agent.remember(state, action, reward, next_state, done)

        # Test replay
        self.agent.replay()
        self.assertGreaterEqual(len(self.agent.memory.memory), self.batch_size,
                                "La mémoire n'est pas suffisante pour l'échantillonnage")

    def test_update_target_model(self):
        """Test de la mise à jour du modèle cible"""
        before_update = list(self.agent.target_model.parameters())[0].clone()
        self.agent.update_target_model()
        after_update = list(self.agent.target_model.parameters())[0].clone()
        self.assertTrue(torch.equal(before_update, after_update), "Les modèles ne sont pas synchronisés correctement")


if __name__ == "__main__":
    unittest.main()
