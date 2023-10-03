import numpy as np
import numba as nb


def build_b(f_v, rewards, states, actions, lambd):

    b_ = np.zeros([len(states), len(actions)])
    for state in states:
        for action in actions:
            for next_state in states:
                b_[state - 1][action - 1][next_state - 1] = rewards[state - 1][action - 1][next_state - 1] + lambd * \
                                                            f_v[next_state - 1]
    return b_


@nb.njit
def check_sum_to_one(array):
    if abs(np.sum(array) - 1) > 1e-6:
        print('error: sum of array is not 1', np.sum(array))
    else:
        print('sum of array is 1')


@nb.njit(nb.float64[:, :, :](nb.float64[:], nb.float64[:, :, :], nb.int64[:], nb.int64[:], nb.float64))
def build_b_sa(F_v, rewards, states, actions, Lambd):
    b_sa = np.zeros((len(states), len(actions), len(states)))
    F_v_lambda = F_v * Lambd
    for i in range(len(states)):
        for j in range(len(actions)):
            for k in range(len(states)):
                b_sa[i][j][k] = rewards[i][j][k] + F_v_lambda[k]
    return b_sa


