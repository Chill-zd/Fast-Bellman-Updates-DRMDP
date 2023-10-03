# author: Yu Zhuodong
import numpy as np
import numba as nb
from numba import float64, int32, int64


# random r_sa


@nb.njit(float64[:](float64[:], float64[:], float64))
def solve_inner_problem(sorted_p_hat, sorted_b_sa, theta):
    p = np.zeros_like(sorted_p_hat)
    sum = 0
    for i in range(len(sorted_b_sa)):
        p[i] = min(sorted_p_hat[i] + theta, 1)
        sum += p[i]
        if sum >= 1:
            p[i] = 1 - sum + p[i]
            break
    return p


@nb.njit(float64(float64[:, :, :, :], int64, int64[:], float64, float64[:, :], int64))
def solve_infty_fast(sample_transitions, state, actions, theta, b_s, N):
    sum = np.zeros(len(actions))
    for action in actions:
        temp_sum = 0
        b_sa = b_s[action]
        sorted_b_sa_index = np.argsort(b_sa)
        sorted_b_sa = b_sa[sorted_b_sa_index]
        for i in range(N):
            p_sa_bar = sample_transitions[i, state, action, :]
            sorted_p_hat = p_sa_bar[sorted_b_sa_index]
            optimal_p = solve_inner_problem(sorted_p_hat, sorted_b_sa, theta)
            temp_sum += np.dot(np.ascontiguousarray(optimal_p), np.ascontiguousarray(sorted_b_sa))
        temp_sum /= N
        sum[action] = temp_sum
    res = np.max(sum)

    return res
