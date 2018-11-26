import gym
import os
import numpy as np
import time

import matplotlib.pyplot as plt
import sys
sys.path.append("../ml_mdp")
from parameters import ALPHA, GAMMA, EPISODES, MAX_ITERATIONS, EPSILON

# Environment initialization
env = gym.make('FrozenLake-v0')

# Q and rewards
Q = np.zeros((env.observation_space.n, env.action_space.n))
rewards = []
iterations = []

time1 = time.time()

def choose_action(state):
    action = 0
    if np.random.uniform(0, 1) < .15:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

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
        # Decreasing Exploration
        action = np.argmax(Q[current, :] + np.random.randn(1, env.action_space.n) * (1 / float(episode + 1)))
        # Constant Exploration
        # action = choose_action(state=state)
        state, reward, done, info = env.step(action)
        t_reward += reward
        Q[current, action] += ALPHA * (reward + GAMMA * np.max(Q[state, :]) - Q[current, action])

    rewards.append(t_reward)
    iterations.append(i)

# Close environment
env.close()
time2 = time.time()
print 'Frozen Lake Q learning took %0.3f ms' % ((time2-time1)*1000.0)
# Plot results
def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
size = EPISODES / 50
chunks = list(chunk_list(rewards, size))
averages = [sum(chunk) / len(chunk) for chunk in chunks]

plt.plot(range(0, len(rewards), size), averages)
plt.title('Frozen Lake Rewards: decreasing exploration')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.savefig('figs/qlearn_reward_frozen_lake_decr.png')
# plt.show()