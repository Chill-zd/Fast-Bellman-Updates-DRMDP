# author: Yu Zhuodong
import numpy as np
from numba import float64, int64


import numba as nb

from fast_bellman_update.fast_algorithm.trisection_alpha.trisection_alpha_l2 import solve_alpha_trisec_l2, cal_alpha_cum_l2


@nb.njit(float64(float64[:, :]))
def min_min(b_s):
    min_val = np.inf
    for i in range(b_s.shape[0]):
        row_min = np.inf
        for j in range(b_s.shape[1]):
            if b_s[i, j] < row_min:
                row_min = b_s[i, j]
        if row_min < min_val:
            min_val = row_min
    return min_val


@nb.njit()
def max_min_array(b_s):
    min_arr = np.empty(b_s.shape[0], dtype=b_s.dtype)
    for i in range(b_s.shape[0]):
        min_arr[i] = np.min(b_s[i])
    res = np.max(min_arr)
    return res


@nb.njit(float64(float64[:, :, :, :], int64, int64[:], float64, float64[:, :], int64))
def solve_gamma_bisection_l2(sample_transitions, state, actions, theta, b_s, N):

    lb = max_min_array(b_s)
    ub = 0
    for i in range(N):
        for action in range(len(actions)):
            ub += np.sum(sample_transitions[i][state][action] * b_s[action])
    ub = ub
    ep = 1e-5
    while abs(ub - lb) > ep:
        mid = (lb + ub) / 2
        temp_sum = np.zeros(len(actions))
        for action in actions:
            p_sa_bar = sample_transitions[:, state, action, :]
            b_sa = b_s[action]
            min_ba = min_min(b_s)
            alpha = solve_alpha_trisec_l2(mid, p_sa_bar, N, b_sa,
                                          min_ba)
            temp_sum[action] = cal_alpha_cum_l2(p_sa_bar, b_sa, N, alpha, mid)
        if np.sum(temp_sum) <= theta * theta:
            ub = mid
        else:
            lb = mid
    return (lb + ub) / 2
