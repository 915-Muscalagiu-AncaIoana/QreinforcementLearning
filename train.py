import time

import gym
import numpy as np

from qagent import Qagent

import gym_toytext

def learn_to_play(agent: Qagent, max_game_steps: int = 10, total_episodes: int = 1000) -> Qagent:
    """
    implementation of the q-learning algorithm, here the q-table values are calculated

    Args:
      max_game_steps (int): number of stepts an agent can take, before the environment is reset
      total_episodes (int): total of training episodes (the number of trials a agent can do)
    """

    rewards = np.zeros(total_episodes)
    epsilons = np.zeros(total_episodes)
    last_states = np.zeros(total_episodes)
    q_averages = np.zeros(total_episodes)

    start = time.time()

    for episode in range(total_episodes):

        state = env.reset()
        game_rewards = 0

        # for each episode loop over the max number of steps that are possible
        # take an action and observe the outcome state (new_state), reward and stopping criterion
        for step in range(max_game_steps):

            action = agent.get_action(state)
            new_state, reward, done, _ = env.step(action)
            agent.update_qtable(state, new_state, action, reward, done)
            state = new_state
            game_rewards += reward

            if done == True:
                break

        rewards[episode] = game_rewards
        last_states[episode] = state
        epsilons[episode] = agent.epsilon
        q_averages[episode] = np.sum(agent.qtable)

        # reduce epsilon, for exploration-exploitation tradeoff
        agent.update_epsilon(episode)

        if episode % 300 == 0:
            elapsed_time = round((time.time() - start), 1)
            print(f"elapsed time [sec]: {elapsed_time}, episode: {episode}")

    agent.rewards = rewards
    agent.last_states = last_states
    agent.epsilons = epsilons
    agent.q_averages = q_averages
    return agent


env = gym.make('NChain-v0')
[env.action_space.sample() for _ in range(10)]
[env.observation_space.sample() for _ in range(10)]
action_size = env.action_space.n
state_size = env.observation_space.n

# Set the training parameters
env.env.slip = 0.0  # avoid slipping in on the chain

max_game_steps = 10  # Set number of stepts an agent can take, before the environment is reset,
total_episodes = 1000  # Set total of training episodes (the number of trials a agent can do)
name = 'Smart Agent 1 - the agent explores and takes future rewards into accountt'
color = "orange"

learning_parameters = {
    'learning_rate': 0.8,
    'gamma': 0.9
}
exploration_parameters = {
    'epsilon': 1,
    'max_epsilon': 1,
    'min_epsilon': 0.0,
    'decay_rate': 0.008
}
q_agent_1 = Qagent(action_size, state_size, learning_parameters, exploration_parameters, name, color)
q_agent_1 = learn_to_play(q_agent_1, max_game_steps=max_game_steps, total_episodes=total_episodes)
