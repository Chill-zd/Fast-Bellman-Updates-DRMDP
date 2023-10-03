# author: Yu Zhuodong
import time

import numpy as np
import numba as nb


# from FOM.Julian_l2_Note import solve_inner_l2


@nb.njit
def l2_norm_squared(x):
    return sum([i ** 2 for i in x])


@nb.njit
def cal_l2_given_gamma(gamma, y_prime, y_hat, sigma, h, theta, N, A, state, y_t, t):
    res = 0
    for i in range(N):
        for a in range(A):
            temp_y_s_t = y_prime[i][a]
            temp_y_hat = y_hat[i][a]

            temp = sigma * (temp_y_s_t / sigma + gamma * temp_y_hat - h[a]) / (1 + sigma * gamma)

            proj_temp = euclidean_proj_simplex_julian(temp)

            proj_temp_contig = np.ascontiguousarray(proj_temp)
            h_a_contig = np.ascontiguousarray(h[a] * sigma)

            y_t[t][state][i][a] = proj_temp

            res += (l2_norm_squared(proj_temp - temp_y_s_t)) / 2 + (
                l2_norm_squared(proj_temp - temp_y_hat)) * gamma + np.dot(proj_temp_contig, h_a_contig)
    res = res - N * theta * theta * gamma

    return res


@nb.njit
def cal_l2_given_gamma_y(gamma, y_prime, y_hat, sigma, h, theta, N, A, state, y_t, t):
    for i in range(N):
        for a in range(A):
            temp_y_s_t = y_prime[i][a]
            temp_y_hat = y_hat[i][a]

            temp = sigma * (temp_y_s_t / sigma + gamma * temp_y_hat - h[a]) / (1 + sigma * gamma)

            proj_temp = euclidean_proj_simplex_julian(temp)

            y_t[t][state][i][a] = proj_temp

    return y_t[t][state]


@nb.njit
def coucum(a, b):
    total = 0
    i = a
    while i <= b:
        total += i
        i += 1
    return total


@nb.njit(nb.float64[:](nb.float64[:]))
def euclidean_proj_simplex_julian(v):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n_features + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w


@nb.njit
def first_order_method_l2(epochs, sample_transition, states, actions, rewards, lambd, theta, N):
    A = len(actions)
    S = len(states)
    T = int((epochs) * (epochs + 1) * (2 * epochs + 1) / 6 + 1)

    x_t = np.zeros((T, S, A), dtype=np.float64)

    h = np.zeros((A, S), dtype=np.float64)

    y_t = np.zeros((T, S, N, A, S), dtype=np.float64)

    v = np.zeros((T, S), dtype=np.float64)
    v[1] = np.ones(S, dtype=np.float64)

    for s in range(S):
        x_t[0][s][0] = 1

    y_t[0] = np.zeros((S, N, A, S), dtype=np.float64)
    for s in range(S):
        for i in range(N):
            for a in range(A):
                for s_prime in range(S):
                    y_t[0][s][i][a][s_prime] = 1 / S

    for epoch in range(epochs):

        epoch = epoch + 1

        tau = 1 / (np.sqrt(A) * lambd * np.linalg.norm(v[epoch]))

        sigma = N * np.sqrt(A) / (lambd * np.linalg.norm(v[epoch]))

        tau_l = int((epoch - 1) * (epoch) * (2 * epoch - 1) / 6)
        T_l = int(epoch * epoch)

        for state in range(1):

            for t in range(tau_l, tau_l + T_l):
                c = np.zeros((S, A), dtype=np.float64)

                temp_x_s_t = x_t[t][state]

                for a in range(A):
                    temp_res = 0
                    for s_prime in range(len(states)):
                        for i in range(N):
                            temp_res += -y_t[t][state][i][a][s_prime] * lambd * v[epoch][s_prime] / N

                    c[state][a] = -rewards[state][a][0] + temp_res

                x_t[t + 1][state] = euclidean_proj_simplex_julian(temp_x_s_t - tau * c[state])

                for a in range(A):
                    for s_prime in range(S):
                        h[a][s_prime] = - lambd / N * (2 * x_t[t + 1][state][a] - temp_x_s_t[a]) * -1 * v[epoch][
                            s_prime]

                lb = 0
                ub = 100
                eps = 1e-3
                function_values = np.empty(2, dtype=np.float64)

                while abs(ub - lb) > eps:
                    left_third = lb + (ub - lb) * 0.382
                    right_third = ub - (ub - lb) * 0.382
                    function_values[0] = cal_l2_given_gamma(left_third, y_t[t][state],
                                                            sample_transition[:, state, :, :], sigma, h, theta, N, A,
                                                            state, y_t, t)
                    function_values[1] = cal_l2_given_gamma(right_third, y_t[t][state],
                                                            sample_transition[:, state, :, :], sigma, h, theta, N, A,
                                                            state, y_t, t)

                    if function_values[0] > function_values[1]:
                        ub = right_third

                    elif function_values[0] < function_values[1]:
                        lb = left_third

                gamma = (lb + ub) / 2

                y_t[t + 1][state] = cal_l2_given_gamma_y(gamma, y_t[t][state], sample_transition[:, state, :, :], sigma,
                                                         h, theta, N, A, state, y_t, t)

        S_l = 0
        for i in range(tau_l + 1, T_l + tau_l + 1):
            S_l += i

        temp_x = np.zeros_like(x_t[0])

        temp_y = np.zeros_like(y_t[0])
        for t in range(tau_l + 1, tau_l + T_l + 1):
            for state in range(S):
                temp_x[state] += t / S_l * x_t[t][state]

        for t in range(tau_l + 1, tau_l + T_l + 1):
            for state in range(S):
                for i in range(N):
                    temp_y[state][i] += t / S_l * y_t[t][state][i]

        sum_temp_y = np.zeros((S, A, S), dtype=np.float64)
        for state in range(S):
            for i in range(N):
                sum_temp_y[state] += 1 / N * temp_y[state][i]

        for state in range(S):
            res = 0
            for a in range(A):
                res += -temp_x[state][a] * rewards[state][a][0]
                for s_prime in range(S):
                    res += temp_x[state][a] * (sum_temp_y[state][a][s_prime] * lambd * -1 * v[epoch][s_prime])
            v[epoch + 1][state] = -res

    return 0


@nb.njit
def first_order_method_l2_VI(sample_transition, states, actions, rewards, lambd, theta, N, delta, max_iter):
    final_v = np.zeros((len(states)), dtype=np.float64)
    epochs = max_iter
    A = len(actions)
    S = len(states)
    T = int((epochs) * (epochs + 1) * (2 * epochs + 1) / 6 + 1)

    x_t = np.zeros((T, S, A), dtype=np.float64)

    h = np.zeros((A, S), dtype=np.float64)

    y_t = np.zeros((T, S, N, A, S), dtype=np.float64)

    v = np.zeros((T, S), dtype=np.float64)
    v[1] = np.ones(S, dtype=np.float64)

    for s in range(S):
        x_t[0][s][0] = 1

    y_t[0] = np.zeros((S, N, A, S), dtype=np.float64)
    for s in range(S):
        for i in range(N):
            for a in range(A):
                for s_prime in range(S):
                    y_t[0][s][i][a][s_prime] = 1 / S

    for epoch in range(epochs):

        epoch = epoch + 1

        tau = 1 / (np.sqrt(A) * lambd * np.linalg.norm(v[epoch]))

        sigma = N * np.sqrt(A) / (lambd * np.linalg.norm(v[epoch]))

        tau_l = int((epoch - 1) * (epoch) * (2 * epoch - 1) / 6)
        T_l = int(epoch * epoch)

        for state in range(len(states)):

            for t in range(tau_l, tau_l + T_l):
                c = np.zeros((S, A), dtype=np.float64)

                temp_x_s_t = x_t[t][state]

                for a in range(A):
                    temp_res = 0
                    for s_prime in range(len(states)):
                        for i in range(N):
                            temp_res += -y_t[t][state][i][a][s_prime] * lambd * v[epoch][s_prime] / N

                    c[state][a] = -rewards[state][a][0] + temp_res

                x_t[t + 1][state] = euclidean_proj_simplex_julian(temp_x_s_t - tau * c[state])

                for a in range(A):
                    for s_prime in range(S):
                        h[a][s_prime] = - lambd / N * (2 * x_t[t + 1][state][a] - temp_x_s_t[a]) * -1 * v[epoch][
                            s_prime]

                lb = 0
                ub = 100
                eps = 1e-6
                function_values = np.empty(2, dtype=np.float64)

                while abs(ub - lb) > eps:
                    left_third = lb + (ub - lb) * 0.382
                    right_third = ub - (ub - lb) * 0.382
                    function_values[0] = cal_l2_given_gamma(left_third, y_t[t][state],
                                                            sample_transition[:, state, :, :], sigma, h, theta, N, A,
                                                            state, y_t, t)
                    function_values[1] = cal_l2_given_gamma(right_third, y_t[t][state],
                                                            sample_transition[:, state, :, :], sigma, h, theta, N, A,
                                                            state, y_t, t)

                    if function_values[0] > function_values[1]:
                        ub = right_third

                    elif function_values[0] < function_values[1]:
                        lb = left_third

                gamma = (lb + ub) / 2

                y_t[t + 1][state] = cal_l2_given_gamma_y(gamma, y_t[t][state], sample_transition[:, state, :, :], sigma,
                                                         h, theta, N, A, state, y_t, t)
        # S_l = sum_t=1^T t
        S_l = 0
        for i in range(tau_l + 1, T_l + tau_l + 1):
            S_l += i

        temp_x = np.zeros_like(x_t[0])

        temp_y = np.zeros_like(y_t[0])
        for t in range(tau_l + 1, tau_l + T_l + 1):
            for state in range(S):
                temp_x[state] += t / S_l * x_t[t][state]

        for t in range(tau_l + 1, tau_l + T_l + 1):
            for state in range(S):
                for i in range(N):
                    temp_y[state][i] += t / S_l * y_t[t][state][i]

        sum_temp_y = np.zeros((S, A, S), dtype=np.float64)
        for state in range(S):
            for i in range(N):
                sum_temp_y[state] += 1 / N * temp_y[state][i]

        for state in range(S):
            res = 0
            for a in range(A):
                res += -temp_x[state][a] * rewards[state][a][0]
                for s_prime in range(S):
                    res += temp_x[state][a] * (sum_temp_y[state][a][s_prime] * lambd * -1 * v[epoch][s_prime])
            v[epoch + 1][state] = -res

        delta_F_v = np.max(np.abs(v[epoch + 1] - v[epoch]))
        if delta_F_v < delta:
            final_v = v[epoch + 1]
            break

    return final_v
