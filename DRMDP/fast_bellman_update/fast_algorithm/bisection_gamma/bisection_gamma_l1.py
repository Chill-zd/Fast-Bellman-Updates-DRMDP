# author: Yu Zhuodong

import numpy as np

import numba as nb

from fast_bellman_update.fast_algorithm.trisection_alpha.trisection_alpha_l1 import solve_alpha_trisec_l1, cal_alpha_cum


@nb.njit()
def max_min_array(b_s):
    min_arr = np.empty(b_s.shape[0], dtype=b_s.dtype)
    for i in range(b_s.shape[0]):
        min_arr[i] = np.min(b_s[i])
    res = np.max(min_arr)
    return res


def solve_gamma_bisection_l1(sample_transitions, state, actions, theta, b_s, N):
    lb = max_min_array(b_s)
    ub = 0
    for i in range(N):
        for action in range(len(actions)):
            ub += np.dot(sample_transitions[i][state][action], b_s[action])
    ub = ub
    if ub < lb:
        print('bisection error')
    ep = 1e-5

    while np.abs(ub - lb) > ep:
        mid = (lb + ub) / 2
        temp_sum = np.zeros(len(actions))
        for action in actions:
            p_sa_bar = sample_transitions[:, state, action, :]
            b_sa = b_s[action]
            alpha = solve_alpha_trisec_l1(mid, p_sa_bar, N, b_sa,
                                          b_s)
            temp_sum[action] = cal_alpha_cum(p_sa_bar, b_sa, N, alpha, mid)
        if sum(temp_sum) <= theta:
            ub = mid
        if sum(temp_sum) > theta:
            lb = mid
    return (lb + ub) / 2
