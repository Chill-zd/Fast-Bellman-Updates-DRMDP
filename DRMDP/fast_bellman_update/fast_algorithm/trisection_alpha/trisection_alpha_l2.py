# author: Yu Zhuodong
import numpy as np
import numba as nb
from numba import float64, int32, int64, float32


@nb.njit(float64(float64[:, :], float64[:], float64, int32, float64))
def sub_solve_alpha_ub(p_sa, b_sa, gamma, N, min_ba):
    index = np.argmin(b_sa)
    temp_max = -np.infty
    e_std = np.zeros(len(b_sa))
    e_std[index] = 1
    for i in nb.prange(N):
        temp = np.abs(e_std - p_sa[i]).sum() / (gamma - min_ba)
        if temp > temp_max:
            temp_max = temp
    return temp_max


@nb.njit(float64[:](float64[:]))
def euclidean_proj_simplex(v):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n_features + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w


@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64))
def solve_sub_l2(p_sa_bar, b_sa, alpha):

    temp_y = p_sa_bar - 0.5 * alpha * b_sa

    optimal_p = euclidean_proj_simplex(temp_y)

    diff = optimal_p - p_sa_bar
    cont_optimal_p = np.ascontiguousarray(optimal_p)
    cont_b_sa = np.ascontiguousarray(b_sa)
    res = np.vdot(diff, diff) + alpha * np.vdot(cont_optimal_p, cont_b_sa)

    return res


@nb.njit(float64(float64[:, :], float64[:], int32, float64, float64))
def cal_alpha_cum_l2(p_sa_bar, b_sa, N, alpha, gamma):
    sum = 0

    for i in range(N):
        temp_p_sa_bar = p_sa_bar[i]

        sum += solve_sub_l2(temp_p_sa_bar, b_sa, alpha)

    sum = sum / N - alpha * gamma

    return sum


@nb.njit(nb.float64(nb.float64, nb.float64[:, :], nb.int64, nb.float64[:], nb.float64))
def solve_alpha_trisec_l2(gamma, p_sa_bar, N, b_sa, min_ba):

    lb = 0.0
    ub = sub_solve_alpha_ub(p_sa_bar, b_sa, gamma, N, min_ba)
    eps = 1e-5

    function_values = np.empty(2, dtype=np.float64)
    left_third = 0.0
    right_third = 0.0

    while abs(ub - lb) > eps:
        left_third = lb + (ub - lb) * 0.382
        right_third = ub - (ub - lb) * 0.382

        function_values[0] = cal_alpha_cum_l2(p_sa_bar, b_sa, N, left_third, gamma)
        function_values[1] = cal_alpha_cum_l2(p_sa_bar, b_sa, N, right_third, gamma)

        index_min = function_values[1] < function_values[0] and 1 or 0

        lb = index_min == 0 and left_third or lb
        ub = index_min == 1 and right_third or ub

    return (lb + ub) / 2
