import numpy as np
import random
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from .ddpg_model import Actor, Critic
from ..utils.obs import Observations
from ..utils.replay_buffer import ReplayBuffer
from ..utils.hyperparams import BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, \
                         LR_CRITIC, WEIGHT_DECAY, UPDATE_EVERY, UPDATE_AMOUNT
from .. import BaseAgent
from ... import characters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent(BaseAgent):
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
        super(DDPGAgent, self).__init__(character)

        self.name = self.__class__.__name__
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Load saved and already trained models if available
        if load_model:
            actor_path = self.model_path('checkpoint_actor.pth')
            critic_path = self.model_path('checkpoint_critic.pth')
            self.actor_local.load_state_dict(torch.load(actor_path,
                                                        map_location=device))
            self.actor_local.eval()
            self.critic_local.load_state_dict(torch.load(critic_path,
                                                         map_location=device))
            self.critic_local.eval()

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step for next learning update
        self.t_step = 0

    def act(self, obs, action_space):
        """Returns actions for given state as per current policy."""
        state = Observations.process(obs, device)
        self.actor_local.eval()
        with torch.no_grad():
            action_probs = self.actor_local(state).cpu().data.numpy()
            action = np.argmax(action_probs)
        self.actor_local.train()
        return [action, action_probs]
        # return action

    def step(self, obs, action, reward, next_obs, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        state = Observations.process(obs, device)
        next_state = Observations.process(next_obs, device)
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                # Number of times to learn and update the network
                for _ in range(UPDATE_AMOUNT):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def model_path(self, file):
        file_path = os.path.realpath(__file__)
        dir_path, _ = os.path.split(file_path)
        model_path = dir_path + f"/models/{file}"
        return model_path
