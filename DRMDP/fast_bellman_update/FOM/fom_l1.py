# author: Yu Zhuodong

import numpy as np
import numba as nb
from numba import jit


@nb.njit
def count_nonequal(arr1, arr2):
    count = 0
    for val in range(len(arr1)):
        # if arr1[val] != arr2[val]:
        if abs(arr1[val] - arr2[val]) > 1e-6:
            count += 1
    return count


@nb.njit
def cal_inner_p(y_prime, y_hat, h, alpha, gamma, action, y, sigma, N):
    res = 0
    for s in range(len(y_prime)):
        res += (h[s] + alpha) * y[s] + (y[s] - y_prime[s]) * (y[s] - y_prime[s]) / (2 * sigma) + gamma * abs(
            y[s] - y_hat[s])
    res = res - alpha
    return res


@nb.njit
def find_optimal_y(alpha, y_prime_list, hat_y_list, h_list, gamma, sigma):
    y = np.zeros_like(y_prime_list)
    for s_prime in range(len(y_prime_list)):
        y_prime = y_prime_list[s_prime]
        hat_y = hat_y_list[s_prime]
        h = h_list[s_prime]

        if (1 / sigma) * (y_prime - hat_y) - h - alpha > gamma:
            y[s_prime] = y_prime - sigma * (h + alpha + gamma)
        if abs((1 / sigma) * (y_prime - hat_y) - h - alpha) <= gamma:
            y[s_prime] = hat_y
        if (1 / sigma) * (y_prime - hat_y) - h - alpha < -gamma:
            y[s_prime] = max(0, y_prime - sigma * (h + alpha - gamma))

    return y


@nb.njit
def build_breakpoints(y_prime_list, hat_y_list, h_list, gamma, sigma):
    candidate_breakpoints = []

    for s in range(len(y_prime_list)):
        y_prime = y_prime_list[s]
        hat_y = hat_y_list[s]
        h = h_list[s]

        alpha1 = (1 / sigma) * y_prime - h + gamma
        alpha2 = (1 / sigma) * (y_prime - hat_y) - h + gamma
        alpha3 = (1 / sigma) * (y_prime - hat_y) - h - gamma

        candidate_breakpoints.extend([alpha1, alpha2, alpha3])

    breakpoints = sorted(list(set(candidate_breakpoints)), reverse=True)

    breakpoints = np.array(breakpoints)

    return breakpoints


@nb.njit
def find_optimal_alpha(breakpoints, y_prime_list, hat_y_list, h_list, gamma, sigma):
    alpha1 = breakpoints[0]
    alpha_star = 0
    old_y = np.zeros_like(y_prime_list)
    sum_y = 0
    for i in range(1, len(breakpoints)):

        y = np.zeros_like(y_prime_list)
        alpha2 = breakpoints[i]

        for s_prime in range(len(y_prime_list)):
            y_prime = y_prime_list[s_prime]
            hat_y = hat_y_list[s_prime]
            h = h_list[s_prime]

            if (1 / sigma) * (y_prime - hat_y) - h - alpha2 > gamma:
                y[s_prime] = y_prime - sigma * (h + alpha2 + gamma)
            if abs((1 / sigma) * (y_prime - hat_y) - h - alpha2) <= gamma:
                y[s_prime] = hat_y
            if (1 / sigma) * (y_prime - hat_y) - h - alpha2 < -gamma:
                y[s_prime] = max(0, y_prime - sigma * (h + alpha2 - gamma))

        # update sum_y
        sum_y = np.sum(old_y)  # sum_y_{alpha1}

        num_active = count_nonequal(old_y, y)

        sum_y += num_active * (alpha1 - alpha2) * sigma

        if sum_y >= 1:
            if abs(sum_y - 1) < 1e-4:
                alpha_star = alpha2
                break
            else:
                alpha_star = alpha2 + (sum_y - 1) / (num_active * sigma)
                break

        else:
            #
            alpha1 = alpha2

            old_y = y

    return alpha_star


@nb.njit
def cal_l1_alpha(gamma, y_prime, y_hat, sigma, h, theta, N, A):
    res = 0
    for i in range(N):
        for a in range(A):
            temp_y_s_t = y_prime[i][a]
            temp_y_hat = y_hat[i][a]

            breakpoints = build_breakpoints(temp_y_s_t, temp_y_hat, h[a], gamma, sigma)

            temp_alpha = find_optimal_alpha(breakpoints, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)

            temp_y = find_optimal_y(temp_alpha, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)

            res += cal_inner_p(temp_y_s_t, temp_y_hat, h[a], temp_alpha, gamma, a, temp_y, sigma, N)

    return res


@nb.njit(nb.float64[:](nb.float64[:]))
def euclidean_proj_simplex_julian(v):
    """Compute the Euclidean projection onto the probability simplex
    for the input vector v"""
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n_features + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w


@nb.njit
def cal_l1_given_gamma(gamma, y_prime, y_hat, sigma, h, theta, N, A):
    res = 0
    for i in range(N):
        for a in range(A):
            temp_y_s_t = y_prime[i][a]
            temp_y_hat = y_hat[i][a]

            breakpoints = build_breakpoints(temp_y_s_t, temp_y_hat, h[a], gamma, sigma)

            temp_alpha = find_optimal_alpha(breakpoints, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)

            temp_y = find_optimal_y(temp_alpha, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)

            res += cal_inner_p(temp_y_s_t, temp_y_hat, h[a], temp_alpha, gamma, a, temp_y, sigma, N)
    res -= theta * N * gamma

    return res


@nb.njit
def cal_l1_given_gamma_y(gamma, y_prime, y_hat, sigma, h, N, A, state, y_t, t):
    for i in range(N):
        for a in range(A):
            temp_y_s_t = y_prime[i][a]
            temp_y_hat = y_hat[i][a]

            breakpoints = build_breakpoints(temp_y_s_t, temp_y_hat, h[a], gamma, sigma)

            temp_alpha = find_optimal_alpha(breakpoints, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)

            temp_y = find_optimal_y(temp_alpha, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)

            y_t[t + 1][state][i][a] = temp_y

    return y_t[t + 1][state]


@nb.njit
def first_order_method_l1(epochs, sample_transition, states, actions, rewards, lambd, theta, N):
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

            # for state in states:
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

                while np.abs(ub - lb) > eps:
                    left_third = lb + (ub - lb) * 0.382
                    right_third = ub - (ub - lb) * 0.382
                    function_values[0] = cal_l1_given_gamma(left_third, y_t[t][state],
                                                            sample_transition[:, state, :, :], sigma, h, theta, N, A)

                    function_values[1] = cal_l1_given_gamma(right_third, y_t[t][state],
                                                            sample_transition[:, state, :, :], sigma, h, theta, N, A)

                    if function_values[0] > function_values[1]:
                        ub = right_third

                    elif function_values[0] < function_values[1]:
                        lb = left_third

                gamma = (lb + ub) / 2

                y_t[t + 1][state] = cal_l1_given_gamma_y(gamma, y_t[t][state], sample_transition[:, state, :, :], sigma,
                                                         h, N, A, state, y_t, t)

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


def first_order_method_l1_VI(sample_transition, states, actions, rewards, lambd, theta, N, delta, max_iter):
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
                eps = 1e-3
                function_values = np.empty(2, dtype=np.float64)

                while np.abs(ub - lb) > eps:
                    left_third = lb + (ub - lb) * 0.382
                    right_third = ub - (ub - lb) * 0.382
                    function_values[0] = cal_l1_given_gamma(left_third, y_t[t][state],
                                                            sample_transition[:, state, :, :], sigma, h, theta, N, A)

                    function_values[1] = cal_l1_given_gamma(right_third, y_t[t][state],
                                                            sample_transition[:, state, :, :], sigma, h, theta, N, A)
                    # print(function_values[0], function_values[1])
                    if function_values[0] > function_values[1]:
                        ub = right_third

                    elif function_values[0] < function_values[1]:
                        lb = left_third

                gamma = (lb + ub) / 2

                y_t[t + 1][state] = cal_l1_given_gamma_y(gamma, y_t[t][state], sample_transition[:, state, :, :], sigma,
                                                         h, N, A, state, y_t, t)

        S_l = 0
        for i in range(tau_l + 1, T_l + tau_l + 1):
            S_l += i

        temp_x = np.zeros_like(x_t[0])
        # temp_y  SxNxAxS
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
