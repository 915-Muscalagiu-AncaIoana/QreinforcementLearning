import numpy as np
import gym

env = gym.make("Taxi-v3")
env.reset()
env.render()
action_size = 6
state_size = 500
V = dict()

# initially the value function for all states
# will be random values close to zero
for i in range(state_size):
    V[i] = np.random.random()

# will take random action for the first time
first_time = True
small_change = 1e-20
gamma = 0.9
episodes = 0
max_episodes = 50000

# generate random policy
policy = dict()
for s in range(state_size):
    policy[s] = env.action_space.sample()

while episodes < max_episodes:
    # policy evaluation
    while True:
        episodes += 1
        if episodes % 100 == 0:
            print("Current episode: {}".format(episodes))
        biggest_change = 0
        # loop through every state present
        for state in range(state_size):
            old_V = V[state]
            # take random action according to policy
            action = policy[state]
            prob, new_state, reward, done = env.env.P[state][action][0]
            V[state] = reward + gamma * V[new_state]
            biggest_change = max(biggest_change, abs(V[state] - old_V))
        if biggest_change < small_change:
            break
        # policy improvement
        policy_changed = False
        for state in range(state_size):
            best_val = -np.inf
            best_action = -1
            for action in range(action_size):
                prob, new_state, reward, done = env.env.P[state][action][0]
                future_reward = reward + gamma * V[new_state]
                if future_reward > best_val:
                    best_val = future_reward
                    best_action = action
            assert best_action != -1
            if policy[state] != best_action:
                policy_changed = True
            policy[state] = best_action

        if not policy_changed:
            break
    print("Total episodes trained: {}".format(episodes))

# play the game
env.reset()
rewards = []

test_episodes = 100
for episode in range(test_episodes):
    state = env.reset()
    total_rewards = 0
    print("*" * 100)
    print("Episode {}".format(episode))
    for step in range(25):
        env.render()
        # Take action which has the highest q value
        # in the current state
        action = policy[state]
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        if done:
            rewards.append(total_rewards)
            print("Score", total_rewards)
            break
        state = new_state
env.close()
print("Average Score", sum(rewards) / test_episodes)