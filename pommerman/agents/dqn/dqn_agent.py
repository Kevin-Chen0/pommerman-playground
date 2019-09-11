import numpy as np
import random
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from .dqn_model import QNetwork
from ..utils.obs import Observations
from ..utils.replay_buffer import ReplayBuffer
from ..utils.hyperparams import BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, UPDATE_EVERY, \
                                EPS_START, EPS_END, EPS_DECAY, LEARNING_RATE
from .. import BaseAgent
from ... import characters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent(BaseAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, load_model=False,
                 character=characters.Bomber):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super(DQNAgent, self).__init__(character)

        self.name = self.__class__.__name__
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Epsilon-greedy Policy
        self.eps = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        if load_model:
            path = self.model_path('checkpoint.pth')
            self.qnetwork_local.load_state_dict(torch.load(path, map_location=device))
            self.qnetwork_local.eval()

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),
                                    lr=LEARNING_RATE)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def act(self, obs, action_space):
        """Returns actions for given state as per current policy."""
        state = Observations.process(obs, device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > self.eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        return action

    def step(self, obs, action, reward, next_obs, done):
        # Save experience in replay memory
        state = Observations.process(obs, device)
        next_state = Observations.process(next_obs, device)
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.long())
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def update_epsilon(self):
        self.eps = max(self.eps_end, self.eps_decay*self.eps)

    def model_path(self, file):
        file_path = os.path.realpath(__file__)
        dir_path, _ = os.path.split(file_path)
        model_path = dir_path + f"/models/{file}"
        return model_path
