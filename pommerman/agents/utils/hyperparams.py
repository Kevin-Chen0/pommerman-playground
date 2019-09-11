# General
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.1              # for soft update of target parameters
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2        # how often to update the network
UPDATE_AMOUNT = 1       # amount of learning updates to the network
# Deep-Q Network (DQN)
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.998
LEARNING_RATE = 5e-4
# Deep Deterministic Policy Gradient (DDPG)
LR_ACTOR = 2e-4         # learning rate of the actor
LR_CRITIC = 2e-4        # learning rate of the critic
# Twin-Delayed DDPG (TD3)
# TAU = 0.01            # Trick: More slowly updating target network
# UPDATE_EVERY = 4      # Trick: Extra delaying the policy update for TD3
