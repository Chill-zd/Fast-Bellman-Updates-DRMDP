# author: Yu Zhuodong

import numpy as np
import numba as nb
from numba import float64, int32, int64


@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64, nb.float64))
def solve_ep_plus_pb(sort_p, sort_b_sa, alpha, pb):
    # sort b_sa

    inner_min = 0 + alpha * pb
    index = 0
    for flag in range(len(sort_b_sa) - 1):
        res = 2 * sort_p[flag] + alpha * (sort_b_sa[-1] - sort_b_sa[flag]) * sort_p[flag]
        # print(sort_p, sort_b_sa)
        # print('p_cumsum', p_cumsum)
        # print(2 * temp + alpha * pb, inner_min)
        if res <= 0:
            inner_min += res
            continue
        else:
            index = flag - 1
            break
    '''
    if index == -1:
        print('sort_p no change', sort_p)
    else:
        # Add the value of first k elements to the last one
        sort_p[-1] += np.sum(sort_p[:index])

        # Set the first k elements to 0
        sort_p[:index] = np.zeros(index)
        # print(sort_b_sa)
        # print('inner_min',inner_min)
        print('sort_p', sort_p)
    '''
    return inner_min


@nb.njit(float64(float64[:, :], float64[:], float64, int32, float64))
def sub_solve_alpha_ub(p_sa, b_sa, gamma, N, min_ba):
    index = np.argmin(b_sa)
    # min_ba = np.min(np.min(b_s, axis=1))
    temp_max = -np.infty
    e_std = np.zeros(len(b_sa))
    e_std[index] = 1
    for i in nb.prange(N):
        temp = np.abs(e_std - p_sa[i]).sum() / (gamma - min_ba)
        if temp > temp_max:
            temp_max = temp
    return temp_max


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


@nb.njit(float64(float64[:, :], float64[:], int32, float64, float64))
def cal_alpha_cum(p_sa_bar, b_sa, N, alpha, gamma):
    sum = 0
    sorted_indices = np.argsort(b_sa)[::-1]

    sort_b_sa = b_sa[sorted_indices]

    for i in range(N):
        temp_p_sa_bar = p_sa_bar[i]
        sort_p = temp_p_sa_bar[sorted_indices]
        pb = np.dot(sort_b_sa, sort_p)
        '''
        if abs(solve_ep_plus_pb(sort_p, sort_b_sa, alpha, pb) - solve_ep_plus_pb_gu(sort_p, sort_b_sa, alpha)) > 1e-6:
            print('a', solve_ep_plus_pb(sort_p, sort_b_sa, alpha, pb))
            print('b', solve_ep_plus_pb_gu(sort_p, sort_b_sa, alpha))
        '''
        # sum += solve_ep_plus_pb_gu(sort_p, sort_b_sa, alpha)
        sum += solve_ep_plus_pb(sort_p, sort_b_sa, alpha, pb)
    # print(sum)
    sum = sum / N - alpha * gamma
    # p_res = np.array(p_res)
    return sum


@nb.njit(int32(float64[:]))
def argmin_1d(a):
    if a[0] > a[1]:
        index = 1
    elif a[0] < a[1]:
        index = 0
    return index


@nb.njit(float64(float64, float64[:, :], int64, float64[:], float64[:, :]))
def solve_alpha_trisec_l1(gamma, p_sa_bar, N, b_sa, b_s):
    # build lower bound and upper bound for alpha:
    lb = 0
    # alpha_ub = 3.5
    min_ba = min_min(b_s)
    '''
    min_ba = np.min(np.min(b_s, axis=1))
    '''
    ub = sub_solve_alpha_ub(p_sa_bar, b_sa, gamma, N, min_ba)
    ub = ub
    # print('lb', lb, 'ub', ub)
    eps = 1e-5
    # print('alpha_ub',alpha_ub)
    # build function for trisection
    # @jit(nopython=True)

    while abs(ub - lb) > eps:
        left_third = lb + (ub - lb) * 0.382
        right_third = ub - (ub - lb) * 0.382

        # Calculate function values for each third
        function_values = np.array([cal_alpha_cum(p_sa_bar, b_sa, N, x, gamma) for x in
                                    [left_third, right_third]])
        # print(function_values.shape[0])
        # Find index corresponding to lowest function value
        index_min = argmin_1d(function_values)
        if index_min == 0:
            lb = left_third
        elif index_min == 1:
            ub = right_third

    return (lb + ub) / 2
