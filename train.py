import pommerman
from pommerman.agents import DullAgent, SimpleAgent
from pommerman.agents.dqn import DQNAgent
from pommerman.agents.ddpg import DDPGAgent
from pommerman.agents.td3 import TD3Agent
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def train(env, n_episodes=100, max_t=1000):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    # training agent and id
    agent_id = env.training_agent if env.training_agent is not None else 0
    agent = env._agents[agent_id]
    # total_time = 0
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    # eps = eps_start                    # initialize epsilon

    for i_episode in range(1, n_episodes+1):
        states = env.reset()
        score = 0
        for t in range(max_t):
            # env.render()
            actions = env.act(states)
            if isinstance(agent, DQNAgent):
                next_states, rewards, done, info = env.step(actions)
                agent.step(states[agent_id], actions[agent_id], rewards[agent_id],
                           next_states[agent_id], done)
            else:
                actions, action_probs = [actions[0][0], actions[1]], actions[0][1]
                next_states, rewards, done, info = env.step(actions)
                agent.step(states[agent_id], action_probs, rewards[agent_id],
                           next_states[agent_id], done)
            states = next_states
            score += rewards[agent_id]
            if done:
                break
        if isinstance(agent, DQNAgent):
            agent.update_epsilon()
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        # eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAvg Score: {:.2f}'.format(i_episode,
                                                       np.mean(scores_window)
                                                       ), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAvg Score: {:.2f}'.format(i_episode,np.mean(scores_window)))
            if isinstance(agent, DQNAgent):
                qnetwork = agent.qnetwork_local.state_dict()
                qnetwork_path = agent.model_path('checkpoint.pth')
                torch.save(qnetwork, qnetwork_path)
            else:
                actor = agent.actor_local.state_dict()
                critic = agent.critic_local.state_dict()
                actor_path = agent.model_path('checkpoint_actor.pth')
                critic_path = agent.model_path('checkpoint_critic.pth')
                torch.save(actor, actor_path)
                torch.save(critic, critic_path)

        # if np.mean(scores_window) >= 200.0:
        #     print('\nEnv solved in {:d} episodes!\tAvg Score: {:.2f}'.
        #           format(i_episode-100, np.mean(scores_window)))
        #     # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        #     break
    env.close()
    return scores


if __name__ == '__main__':

    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)
    # Pick the registry environment
    REGISTRY = "PommeFFAFast-v0"
    # agent = DQNAgent(network, replay)
    STATE = 396
    ACTION = 6
    SEED = 42
    agent1 = DQNAgent(state_size=STATE, action_size=ACTION, seed=SEED)
    agent2 = DDPGAgent(state_size=STATE, action_size=ACTION, seed=SEED)
    agent3 = TD3Agent(state_size=STATE, action_size=ACTION, seed=SEED)
    # Create a set of agents (up to four)
    agent_list = [
        agent2,
        SimpleAgent()
    ]
    # Initialize environment
    env = pommerman.make(REGISTRY, agent_list)
    type(env)
    # Env info
    env.action_space
    env.observation_space
    env.reward_range
    env.model
    env.training_agent
    # Run
    scores = train(env, n_episodes=300)
    # Close env
    env.close()
    # Plot the scores #########################################################
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores)), pd.DataFrame(scores).rolling(20).mean())
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
