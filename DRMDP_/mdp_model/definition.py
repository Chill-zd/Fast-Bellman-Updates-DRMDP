# author: Yu Zhuodong
import numpy as np


class MDP:
    def __init__(self, states, actions, transitions, rewards, delta):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.delta = delta

    def get_transition_prob(self, state, action, next_state):
        return self.transitions[state][action][next_state]

    def get_reward(self, state, action):
        return self.rewards[state][action]

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def get_delta(self):
        return self.delta


class GarnetMDP(MDP):
    def __init__(self, states, actions, transitions, rewards, delta, nb):
        super().__init__(states, actions, transitions, rewards, delta)
        self.nb = nb

    def get_transition_prob(self, state, action, next_state):
        if next_state in self.transitions[state][action]:
            return self.transitions[state][action][next_state]
        else:
            return 0

    def build_random(self, state_num):
        temp = np.random.rand(state_num)
        temp = temp / np.sum(temp)
        return temp

    def build_transitions(self):
        transitions = np.zeros((len(self.states), len(self.actions), len(self.states)))
        for state in self.states:
            # print(self.states)
            for action in self.actions:
                next_states = np.random.choice(self.states, size=int(len(self.states) * self.nb), replace=False)
                random_p_next_states = self.build_random(len(next_states))
                flag = 0
                for next_state in next_states:
                    transitions[state - 1][action - 1][next_state - 1] = random_p_next_states[flag]
                    flag += 1
        return transitions

    def build_reward(self):
        rewards = np.zeros((len(self.states), len(self.actions), len(self.states)))
        for state in self.states:
            # print(self.states)
            for action in self.actions:
                temp_r = np.random.uniform(0, 10)
                for next_state in self.states:
                    # rewards[state - 1][action - 1][next_state - 1] = np.random.uniform(0, 10)
                    rewards[state - 1][action - 1][next_state - 1] = temp_r
        return rewards

# print(sample_transitions[1])
