# author: Yu Zhuodong
import numpy as np
import gym


class GARNonStationaryEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_states, n_actions, horizon):
        self.n_states = n_states
        self.n_actions = n_actions
        self.horizon = horizon
        self.curr_step = 0
        self.reward_dist = None
        self.transition_dist = None
        self.reset()

    def reset(self):
        self.curr_state = 0
        self.reward_dist = np.random.rand(self.n_states, self.n_actions)
        self.transition_dist = np.random.rand(self.n_states, self.n_actions, self.n_states)
        self.transition_dist /= np.sum(self.transition_dist, axis=2, keepdims=True)
        self.curr_step = 0
        return self.curr_state

    def step(self, action):
        assert self.curr_state is not None, "Please reset the environment before starting."
        reward = self.reward_dist[self.curr_state, action]
        prob = self.transition_dist[self.curr_state, action]
        next_state = np.random.choice(self.n_states, p=prob)
        done = self.curr_step == self.horizon
        self.curr_state = next_state
        self.curr_step += 1
        return self.curr_state, reward, done, {}


import gym

env = GARNonStationaryEnv(n_states=3, n_actions=2, horizon=10)
state = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state

print(f"Total reward: {total_reward}")
