import pommerman
from pommerman.agents import HybridAgent
from pommerman.agents.dqn import DQNAgent
from pommerman.agents.ddpg import DDPGAgent
from pommerman.agents.td3 import TD3Agent
from collections import defaultdict
import numpy as np


def populate(eps):
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)
    # Pick the registry environment
    REGISTRY = "PommeFFAFast-v0"
    # Input the env parameters
    STATE = 396
    ACTION = 6
    SEEDS = [2, 10, 25, 42, 64]
    # Initialize gathering of scores for all agents and for all seeds
    scoring_list = defaultdict(list)
    win_list = defaultdict(list)

    for seed in SEEDS:
        agent1 = DQNAgent(state_size=STATE, action_size=ACTION, seed=seed,
                          load_model=True)
        agent2 = DDPGAgent(state_size=STATE, action_size=ACTION, seed=seed,
                           load_model=True)
        agent3 = TD3Agent(state_size=STATE, action_size=ACTION, seed=seed,
                          load_model=True)
        agents = [agent1, agent2, agent3]
        for agent in agents:
            # Create a set of agents (up to four)
            agent_list = [
                agent,
                HybridAgent(eps=eps)
            ]
            # Initialize environment
            env = pommerman.make(REGISTRY, agent_list)
            type(env)
            # Run env
            scores, wincount = run(env, n_episodes=100, max_t=1000)
            scoring_list[f'{agent.name}'].append(scores)
            win_list[f'{agent.name}'].append(wincount)
        print(f"Seed {seed} has finished running.")

    return scoring_list, win_list


def run(env, n_episodes=100, max_t=1000):
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    agent_id = 0
    agent = env._agents[agent_id]
    scores = []
    win_count = 0
    # Run the episodes just like OpenAI Gym
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        done = False
        score = 0
        for t in range(max_t):
            # env.render()
            actions = env.act(state)
            if not isinstance(agent, DQNAgent):
                actions = [actions[0][0], actions[1]]
            next_state, reward, done, info = env.step(actions)
            state = next_state
            score += reward[agent_id]
            if done:
                if reward[agent_id] == 1:
                    win_count += 1
                break
        scores.append(score)  # save most recent score

    # env.close()
    # print(f"Run finished. Avg score: {round(np.mean(scores), 4)}.")
    return scores, win_count


if __name__ == '__main__':
    # Execute and gather all of the scores
    # Eps: how much randomnness is the HybridAgent
    scoring_list1, win_list1 = populate(eps=0.05)
    print("eps: 0.05")
    print(win_list1)
    scoring_list2, win_list2 = populate(eps=0.1)
    print("eps: 0.1")
    print(win_list2)
    scoring_list3, win_list3 = populate(eps=0.15)
    print("eps: 0.15")
    print(win_list3)
    scoring_list4, win_list4 = populate(eps=0.2)
    print("eps: 0.2")
    print(win_list4)
    scoring_list5, win_list5 = populate(eps=0.25)
    print("eps: 0.25")
    print(win_list5)
    scoring_list6, win_list6 = populate(eps=0.3)
    scoring_list7, win_list7 = populate(eps=0.35)

    print("eps: 0.05")
    win_list1
    print("eps: 0.1")
    win_list2
    print("eps: 0.15")
    win_list3
    print("eps: 0.2")
    win_list4
    print("eps: 0.25")
    win_list5
    print("eps: 0.3")
    win_list6

    print("eps: 0.35")
    win_list7
