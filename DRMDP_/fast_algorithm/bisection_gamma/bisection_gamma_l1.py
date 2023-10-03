# author: Yu Zhuodong

import numpy as np

from fast_algorithm.trisection_alpha.trisection_alpha_l1 import solve_alpha_trisec_l1, cal_alpha_cum


import numba as nb


@nb.njit()
def max_min_array(b_s):
    min_arr = np.empty(b_s.shape[0], dtype=b_s.dtype)
    for i in range(b_s.shape[0]):
        min_arr[i] = np.min(b_s[i])
    res = np.max(min_arr)
    return res


# @nb.njit(nb.float64(nb.float64[:, :, :, :], nb.int64, nb.int64[:], nb.float64, nb.float64[:], nb.int64))
def solve_gamma_bisection_l1(sample_transitions, state, actions, theta, b_s, N):
    # get p_sa_bar and b_sa

    # print(len(sample_transitions))
    # print(len(p_sa_bar))
    # get lower bound and upper bound for gamma
    lb = max_min_array(b_s)
    # lb = np.max(np.min(b_s))
    ub = 0
    for i in range(N):
        for action in range(len(actions)):
            ub += np.dot(sample_transitions[i][state][action], b_s[action])
    ub = ub
    if ub < lb:
        print('bisection error')
    # print('bisection','lb', lb, 'ub', ub)
    # print(lb)
    # print(ub)
    # ub = max_reward / (1 - Lambd)
    ep = 1e-5

    # build function for bisection
    # eps = ep * theta / (2 * len(actions) * ub + len(actions) * ep)

    # print('eps', eps)

    while np.abs(ub - lb) > ep:
        mid = (lb + ub) / 2
        # given state
        # for each gamma,
        # For each action , we need to solve the sub problem for alpha
        temp_sum = np.zeros(len(actions))
        for action in actions:
            # print('action',action)
            p_sa_bar = sample_transitions[:, state, action, :]
            b_sa = b_s[action]
            # print('b_s ',b_s )
            # print('p_sa_bar',p_sa_bar)
            '''
            p_sa_bar = get_N_s_a_S(N, state, action,
                                   sample_transitions)  # contain the probability of p_sa in N samples
            '''
            alpha = solve_alpha_trisec_l1(mid, p_sa_bar, N, b_sa,
                                          b_s)  # solve_alpha_trisec_l1(gamma, p_sa_bar, N, b_sa)
            # print(alpha)
            # print('l1_sum',l1_sum)
            temp_sum[action] = cal_alpha_cum(p_sa_bar, b_sa, N, alpha, mid)
            # print('cal_dis_with_given_lpha(p_sa_bar, b_sa, N, alpha)',cal_dis_with_given_alpha(p_sa_bar, b_sa, N, alpha))
            # print(temp_sum)
        # print(lb,ub, temp_sum)
        if sum(temp_sum) <= theta:
            ub = mid
        if sum(temp_sum) > theta:
            lb = mid
    # print(lb, ub)
    return (lb + ub) / 2
