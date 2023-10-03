#!/usr/bin/env python3.7
import cProfile
from math import sqrt

import numpy as np

from mdp_model import definition
from test_speed import compare_bellman_operator_l1, compare_VI_operator_l1, compare_bellman_operator_l2, \
    compare_VI_operator_l2, compare_bellman_operator_linfty, compare_VI_operator_linfty


def __main__():
    # We can change the parameters here: N, S, A
    '''
    Here we can change parameter N,S,A
    '''
    nStates = 5
    nActions = 5
    N = 5

    '''
    build MDP env
    '''
    n_b = 0.2

    theta = sqrt(n_b * nActions)  # For Garnet MDP, \theta = sqrt(n_b * nActions)
    states = np.arange(0, nStates, 1, dtype=int)
    actions = np.arange(0, nActions, 1, dtype=int)
    transitions = {}
    rewards = {}
    Lambd = 0.8

    sample_transitions = []

    mdp = definition.GarnetMDP(states, actions, transitions, rewards, Lambd, nb=0.2)
    nominal_transition = mdp.build_transitions()
    rewards = mdp.build_reward()

    # build N sample transitions into sample_transitions[N]:
    for i in range(N):
        temp_transition = mdp.build_transitions()
        temp_transition = 0.95 * nominal_transition + 0.05 * temp_transition
        sample_transitions.append(temp_transition)

    sample_transitions = np.array(sample_transitions)

    '''
    Bellman update / Value iteration experiment: 
    Code in test_speed.py
    '''

    # ------------------------------------    l1 norm ---------------------------------------- #
    '''test bellman speed for l_1'''
    print('test bellman speed for l1 \n')
    compare_bellman_operator_l1(sample_transitions, states, actions, rewards, Lambd, theta, N)
    '''test VI speed for l1'''
    print('test VI speed for l1 \n')
    compare_VI_operator_l1(sample_transitions, states, actions, rewards, theta, N)

    # ------------------------------------    l2 norm ---------------------------------------- #

    '''test bellman speed for l_2'''
    print('test bellman speed for l2 \n')
    compare_bellman_operator_l2(sample_transitions, states, actions, rewards, Lambd, theta, N)

    '''test VI speed for l2'''
    print('test VI speed for l2 \n')
    compare_VI_operator_l2(sample_transitions, states, actions, rewards, theta, N)

    # ------------------------------------    linfty norm ---------------------------------------- #

    '''test bellman speed for l_infty'''
    print('test bellman speed for linfty \n')
    compare_bellman_operator_linfty(sample_transitions, states, actions, rewards, Lambd, theta, N)
    '''test VI speed for linfty'''
    print('test VI speed for linfty \n')
    compare_VI_operator_linfty(sample_transitions, states, actions, rewards, theta, N)


if __name__ == '__main__':
    __main__()
