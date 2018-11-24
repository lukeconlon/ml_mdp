import gym
import os
import numpy as np

import matplotlib.pyplot as plt
import sys
sys.path.append("../ml_mdp")
from parameters import ALPHA, GAMMA, EPISODES, MAX_ITERATIONS

# Environment initialization
env = gym.make('Taxi-v2')

# Q and rewards
Q = np.zeros((env.observation_space.n, env.action_space.n))
rewards = []
iterations = []

# Parameters
# alpha = 0.85
# discount = 0.99
# episodes = 1000

# Episodes
for episode in xrange(EPISODES):
    # Refresh state
    state = env.reset()
    done = False
    t_reward = 0
    # max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    # Run episode
    for i in xrange(MAX_ITERATIONS):
        if done:
            break

        current = state
        action = np.argmax(Q[current, :] + np.random.randn(1, env.action_space.n) * (1 / float(episode + 1)))

        state, reward, done, info = env.step(action)
        t_reward += reward
        Q[current, action] += ALPHA * (reward + GAMMA * np.max(Q[state, :]) - Q[current, action])

    rewards.append(t_reward)
    iterations.append(i)

# Close environment
env.close()

# Plot results
def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

size = 5
chunks = list(chunk_list(rewards, size))
averages = [sum(chunk) / len(chunk) for chunk in chunks]

plt.plot(range(0, len(rewards), size), averages)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()