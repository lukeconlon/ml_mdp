import gym
import numpy as np
import time, pickle, os


class QLearner:
    def __init__(self, epsilon=0.9, episodes=10000, max_steps=100, alpha=0.81, gamma=0.975):
        # self.env = gym.make('FrozenLake-v0')
        # self.env = gym.make('FrozenLake8x8-v0')
        self.env = gym.make('Taxi-v2')
        self.epsilon = epsilon
        self.total_episodes = episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.reward = []

    def choose_action(self, state):
        action = 0
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        return action

    def learn(self, state, state2, reward, action):
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = self.Q[state, action] + self.alpha * (target - predict)

def main():
    q_learner = QLearner(gamma=1.0, alpha=0.95)
    all_rewards = []
    # Start
    for episode in range(q_learner.total_episodes):
        state = q_learner.env.reset()
        t = 0
        reward_sum = 0
        while t < q_learner.max_steps:
            # q_learner.env.render()
            action = q_learner.choose_action(state)
            state2, reward, done, info = q_learner.env.step(action)
            q_learner.learn(state, state2, reward, action)
            state = state2
            reward_sum += reward
            t += 1
            if done:
                break
            # time.sleep(0.9)
        all_rewards.append(reward_sum)
    print "Avg. Reward per Episode: " + str(sum(all_rewards)/q_learner.total_episodes)
    print(np.round(q_learner.Q, 2))
    #
    # with open("frozenLake_qTable.pkl", 'wb') as f:
    #     pickle.dump(Q, f)







if __name__ == '__main__':
    main()