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

        # calculate breakpoints for this y_prime, hat_y, h
        alpha1 = (1 / sigma) * y_prime - h + gamma
        alpha2 = (1 / sigma) * (y_prime - hat_y) - h + gamma
        alpha3 = (1 / sigma) * (y_prime - hat_y) - h - gamma

        # add breakpoints to candidate list
        candidate_breakpoints.extend([alpha1, alpha2, alpha3])

    # sort and remove duplicates
    breakpoints = sorted(list(set(candidate_breakpoints)), reverse=True)

    # convert to numpy array
    breakpoints = np.array(breakpoints)
    # print('breakpoints: ', breakpoints)
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

        # update num_active based on whether the variable starts or stops changing at alpha2 for all s_prime
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

            # print('num_active',num_active)
            # cal_num_active(alpha1, check_breakpoints)
            old_y = y

    # calculate optimal alpha using mean value theorem
    # alpha_star = alpha2 - (sum_y - 1) / num_active
    # print('num_active',num_active)

    return alpha_star


@nb.njit
def cal_l1_alpha(gamma, y_prime, y_hat, sigma, h, theta, N, A):
    # print the shape of all inputs of the function:
    # print(y_prime.shape, y_hat.shape,  h.shape,y_t.shape)

    # print('sigma, gamma, y_prime, y_hat, h, theta, N',sigma, gamma, y_prime, y_hat, h, theta, N)
    res = 0
    for i in range(N):
        for a in range(A):
            temp_y_s_t = y_prime[i][a]
            temp_y_hat = y_hat[i][a]

            breakpoints = build_breakpoints(temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # a1, a2 = find_optimal_alpha(breakpoints, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # print('a1,a2',a1,a2)
            # temp_alpha = find_optimal_alpha_1(a1, a2, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            temp_alpha = find_optimal_alpha(breakpoints, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # temp_alpha = find_optimal_alpha_bisection(breakpoints, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # print(temp_alpha1-temp_alpha)
            temp_y = find_optimal_y(temp_alpha, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # print(np.sum(temp_y))
            res += cal_inner_p(temp_y_s_t, temp_y_hat, h[a], temp_alpha, gamma, a, temp_y, sigma, N)
    # print('temp',temp)
    # proj_temp = euclidean_proj_simplex(temp)
    # res = -0.5 * N * theta * theta * gamma + 0.5 * np.linalg.norm(proj_temp - y_hat) ** 2
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
            # a1, a2 = find_optimal_alpha(breakpoints, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # print('a1,a2',a1,a2)
            # temp_alpha = find_optimal_alpha_1(a1, a2, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            temp_alpha = find_optimal_alpha(breakpoints, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # temp_alpha = find_optimal_alpha_bisection(breakpoints, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # print(temp_alpha1-temp_alpha)
            temp_y = find_optimal_y(temp_alpha, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # print(np.sum(temp_y))
            res += cal_inner_p(temp_y_s_t, temp_y_hat, h[a], temp_alpha, gamma, a, temp_y, sigma, N)
    res -= theta * N * gamma
    # print('temp',temp)
    # proj_temp = euclidean_proj_simplex(temp)
    # res = -0.5 * N * theta * theta * gamma + 0.5 * np.linalg.norm(proj_temp - y_hat) ** 2
    return res


# def cal_l1_given_gamma(gamma, y_prime, y_hat, sigma, h, theta, N, A, state, y_t, t):
#     # print the shape of all inputs of the function:
#     # print(y_prime.shape, y_hat.shape,  h.shape,y_t.shape)
#
#     # print('sigma, gamma, y_prime, y_hat, h, theta, N',sigma, gamma, y_prime, y_hat, h, theta, N)
#     S = A
#     res, p = solve_alpha(y_prime, y_hat, sigma, h, gamma, S, N, A, theta)
#
#     res = res - theta * gamma
#
#     # print('temp',temp)
#     # proj_temp = euclidean_proj_simplex(temp)
#     # res = -0.5 * N * theta * theta * gamma + 0.5 * np.linalg.norm(proj_temp - y_hat) ** 2
#     return res


# def cal_l1_given_gamma_y(gamma, y_prime, y_hat, sigma, h, N, A, state, y_t, t):
#     # print('sigma, gamma, y_prime, y_hat, h, theta, N',sigma, gamma, y_prime, y_hat, h, theta, N)
#
#     S = A
#     theta = 0
#     res, p = solve_alpha(y_prime, y_hat, sigma, h, gamma, S, N, A, theta)
#     for i in range(N):
#         for a in range(A):
#             y_t[t + 1][state][i][a] = p[i][a]
#
#     # print('temp',temp)
#     # proj_temp = euclidean_proj_simplex(temp)
#     # res = -0.5 * N * theta * theta * gamma + 0.5 * np.linalg.norm(proj_temp - y_hat) ** 2
#     return y_t[t + 1][state]

@nb.njit
def cal_l1_given_gamma_y(gamma, y_prime, y_hat, sigma, h, N, A, state, y_t, t):
    for i in range(N):
        for a in range(A):
            temp_y_s_t = y_prime[i][a]
            temp_y_hat = y_hat[i][a]

            breakpoints = build_breakpoints(temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # a1, a2 = find_optimal_alpha(breakpoints, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # print('a1,a2',a1,a2)
            # temp_alpha = find_optimal_alpha_1(a1, a2, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            temp_alpha = find_optimal_alpha(breakpoints, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # temp_alpha = find_optimal_alpha_bisection(breakpoints, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # print(temp_alpha1-temp_alpha)
            temp_y = find_optimal_y(temp_alpha, temp_y_s_t, temp_y_hat, h[a], gamma, sigma)
            # print('temp_y',temp_y)
            # temp_y = temp_y / np.sum(temp_y)
            y_t[t + 1][state][i][a] = temp_y

    # print('temp',temp)
    # proj_temp = euclidean_proj_simplex(temp)
    # res = -0.5 * N * theta * theta * gamma + 0.5 * np.linalg.norm(proj_temp - y_hat) ** 2
    return y_t[t + 1][state]


@nb.njit
def first_order_method_l1(epochs, sample_transition, states, actions, rewards, lambd, theta, N):
    A = len(actions)
    S = len(states)
    T = int((epochs) * (epochs + 1) * (2 * epochs + 1) / 6 + 1)
    # print('T', T)
    # define x_t as aTxSxA matrix
    x_t = np.zeros((T, S, A), dtype=np.float64)

    # define temp_h
    h = np.zeros((A, S), dtype=np.float64)

    # define y_t as a TxSxNxAxS matrix
    y_t = np.zeros((T, S, N, A, S), dtype=np.float64)

    # initialize v_1,x_bar_0,y_bar_0:
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
    # definr A as the number of actions

    for epoch in range(epochs):
        # print("epoch is ", epoch)
        epoch = epoch + 1

        tau = 1 / (np.sqrt(A) * lambd * np.linalg.norm(v[epoch]))
        # print("tau is ", tau)
        sigma = N * np.sqrt(A) / (lambd * np.linalg.norm(v[epoch]))
        # print("sigma is ", sigma)

        tau_l = int((epoch - 1) * (epoch) * (2 * epoch - 1) / 6)
        T_l = int(epoch * epoch)

        for state in range(1):

            # for state in states:
            for t in range(tau_l, tau_l + T_l):
                c = np.zeros((S, A), dtype=np.float64)
                # print("t is ", t, tau_l, tau_l + T_l)
                # temp_x_s_t is x_t[0,state,:,:]
                temp_x_s_t = x_t[t][state]
                # c = 1\N \sum_{i=1}^N \sum_{a=1}^A \sum_{s_\prime =1}^{len(states)} y[t][i][state][a][s_prime] (rewards[state][a][s_prime] + lambd* v[s_prime])

                for a in range(A):
                    temp_res = 0
                    for s_prime in range(len(states)):
                        for i in range(N):
                            temp_res += -y_t[t][state][i][a][s_prime] * lambd * v[epoch][s_prime] / N
                            # print(y_t[t][i][state][a][s_prime], rewards[state][a][s_prime], lambd, v[s_prime])
                    c[state][a] = -rewards[state][a][0] + temp_res

                # print("c is ", x_t[t + 1][state].shape)
                # new version
                x_t[t + 1][state] = euclidean_proj_simplex_julian(temp_x_s_t - tau * c[state])
                # old_version
                # x_t[t + 1][state] = euclidean_proj_simplex_julian(temp_x_s_t - tau * c[state])
                # print('x_t[t + 1]', x_t[t + 1][state])
                # update h
                for a in range(A):
                    for s_prime in range(S):
                        # h[a][s_prime] = - lambd / N * (2 * x_t[t + 1][state][a] - temp_x_s_t[a]) * v[epoch][s_prime]
                        h[a][s_prime] = - lambd / N * (2 * x_t[t + 1][state][a] - temp_x_s_t[a]) * -1 * v[epoch][
                            s_prime]
                # print c[state],h
                # print("h is ", h)
                # print("c is ", c[state])
                # trisection gamma:
                lb = 0
                ub = 100
                eps = 1e-6
                function_values = np.empty(2, dtype=np.float64)
                # if epoch == 1:
                #     if state == 0:
                #         np.save('y_prime.npy', y_t[t][state])
                #         np.save('y_hat.npy', sample_transition[:, state, :, :])
                #         np.save('h.npy', h)

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

                # if gamma close to lb or ub, print warning
                # if abs(gamma - lb) < 1e-3 or abs(gamma - ub) < 1e-3:
                # if abs(gamma - store_ub) < 1e-8:
                #     print('warning: gamma is close to ub')
                # elif abs(gamma - store_lb) < 1e-8:
                #     print('warning: gamma is close to lb')
                # else:
                #     print('gamma', gamma)

                # update y_t with optimal gamma
                y_t[t + 1][state] = cal_l1_given_gamma_y(gamma, y_t[t][state], sample_transition[:, state, :, :], sigma,
                                                         h, N, A, state, y_t, t)
                # for i in range(N):
                #     for a in range(A):
                #         print(np.sum(y_t[t + 1][state][i][a]))
        # S_l = sum_t=1^T t
        S_l = 0
        for i in range(tau_l + 1, T_l + tau_l + 1):
            S_l += i
        # print('epoch', epoch, 'S_l', S_l, 'tau_l', tau_l, 'T_l', T_l)

        # print('y_t',y_t[2])
        # y_t TxSxNxAxS
        temp_x = np.zeros_like(x_t[0])
        # temp_y  SxNxAxS
        temp_y = np.zeros_like(y_t[0])
        for t in range(tau_l + 1, tau_l + T_l + 1):
            for state in range(S):
                temp_x[state] += t / S_l * x_t[t][state]
        # print(temp_x[0])
        # for state in range(S):
        #     print('sum',np.sum(temp_x[state]))
        for t in range(tau_l + 1, tau_l + T_l + 1):
            for state in range(S):
                for i in range(N):
                    temp_y[state][i] += t / S_l * y_t[t][state][i]
        # print("temp_x is ", temp_x)
        sum_temp_y = np.zeros((S, A, S), dtype=np.float64)
        for state in range(S):
            for i in range(N):
                sum_temp_y[state] += 1 / N * temp_y[state][i]
        # for state in range(S):
        #     for a in range(A):
        #         print('sum_temp_y',np.sum(sum_temp_y[state][a]))
        # print("sum_temp_y is ", sum_temp_y)
        # update v use temp_x,sum_temp_y

        for state in range(S):
            res = 0
            for a in range(A):
                res += -temp_x[state][a] * rewards[state][a][0]
                for s_prime in range(S):
                    # assert temp_x[state][a] >= 0
                    # assert sum_temp_y[state][a][s_prime] >= 0
                    # print(temp_x[state][a],sum_temp_y[state][a][s_prime])
                    # print(rewards[state][a][s_prime],lambd*v[epoch][s_prime])
                    res += temp_x[state][a] * (sum_temp_y[state][a][s_prime] * lambd * -1 * v[epoch][s_prime])
            v[epoch + 1][state] = -res
        # v = np.ones(S, dtype=np.float64)

    return 0


def first_order_method_l1_VI(sample_transition, states, actions, rewards, lambd, theta, N, delta, max_iter):
    # print('fir', sample_transition.shape, states.shape, actions.shape, rewards.shape)
    # for l = 1,...,k,calculate \sum_{i=1}^{k} i^2 = k(k+1)(2k+1)/6
    # total iteration T = k(k+1)(2k+1)/6
    final_v = np.zeros((len(states)), dtype=np.float64)
    epochs = max_iter
    A = len(actions)
    S = len(states)
    T = int((epochs) * (epochs + 1) * (2 * epochs + 1) / 6 + 1)
    # print('T', T)
    # define x_t as aTxSxA matrix
    x_t = np.zeros((T, S, A), dtype=np.float64)

    # define temp_h
    h = np.zeros((A, S), dtype=np.float64)

    # define y_t as a TxSxNxAxS matrix
    y_t = np.zeros((T, S, N, A, S), dtype=np.float64)

    # initialize v_1,x_bar_0,y_bar_0:
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
    # definr A as the number of actions

    for epoch in range(epochs):
        # print("epoch is ", epoch)
        epoch = epoch + 1

        tau = 1 / (np.sqrt(A) * lambd * np.linalg.norm(v[epoch]))
        # print("tau is ", tau)
        sigma = N * np.sqrt(A) / (lambd * np.linalg.norm(v[epoch]))
        # print("sigma is ", sigma)

        tau_l = int((epoch - 1) * (epoch) * (2 * epoch - 1) / 6)
        T_l = int(epoch * epoch)

        for state in range(len(states)):

            # for state in states:
            for t in range(tau_l, tau_l + T_l):
                c = np.zeros((S, A), dtype=np.float64)
                # print("t is ", t, tau_l, tau_l + T_l)
                # temp_x_s_t is x_t[0,state,:,:]
                temp_x_s_t = x_t[t][state]
                # c = 1\N \sum_{i=1}^N \sum_{a=1}^A \sum_{s_\prime =1}^{len(states)} y[t][i][state][a][s_prime] (rewards[state][a][s_prime] + lambd* v[s_prime])

                for a in range(A):
                    temp_res = 0
                    for s_prime in range(len(states)):
                        for i in range(N):
                            temp_res += -y_t[t][state][i][a][s_prime] * lambd * v[epoch][s_prime] / N
                            # print(y_t[t][i][state][a][s_prime], rewards[state][a][s_prime], lambd, v[s_prime])
                    c[state][a] = -rewards[state][a][0] + temp_res

                # print("c is ", x_t[t + 1][state].shape)
                # new version
                x_t[t + 1][state] = euclidean_proj_simplex_julian(temp_x_s_t - tau * c[state])
                # old_version
                # x_t[t + 1][state] = euclidean_proj_simplex_julian(temp_x_s_t - tau * c[state])
                # print('x_t[t + 1]', x_t[t + 1][state])
                # update h
                for a in range(A):
                    for s_prime in range(S):
                        # h[a][s_prime] = - lambd / N * (2 * x_t[t + 1][state][a] - temp_x_s_t[a]) * v[epoch][s_prime]
                        h[a][s_prime] = - lambd / N * (2 * x_t[t + 1][state][a] - temp_x_s_t[a]) * -1 * v[epoch][
                            s_prime]
                # print c[state],h
                # print("h is ", h)
                # print("c is ", c[state])
                # trisection gamma:
                lb = 0
                ub = 100
                eps = 1e-3
                function_values = np.empty(2, dtype=np.float64)
                # if epoch == 1:
                #     if state == 0:
                #         np.save('y_prime.npy', y_t[t][state])
                #         np.save('y_hat.npy', sample_transition[:, state, :, :])
                #         np.save('h.npy', h)

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

                # if gamma close to lb or ub, print warning
                # if abs(gamma - lb) < 1e-3 or abs(gamma - ub) < 1e-3:
                # if abs(gamma - store_ub) < 1e-8:
                #     print('warning: gamma is close to ub')
                # elif abs(gamma - store_lb) < 1e-8:
                #     print('warning: gamma is close to lb')
                # else:
                #     print('gamma', gamma)

                # update y_t with optimal gamma
                y_t[t + 1][state] = cal_l1_given_gamma_y(gamma, y_t[t][state], sample_transition[:, state, :, :], sigma,
                                                         h, N, A, state, y_t, t)
                # for i in range(N):
                #     for a in range(A):
                #         print(np.sum(y_t[t + 1][state][i][a]))
        # S_l = sum_t=1^T t
        S_l = 0
        for i in range(tau_l + 1, T_l + tau_l + 1):
            S_l += i
        # print('epoch', epoch, 'S_l', S_l, 'tau_l', tau_l, 'T_l', T_l)

        # print('y_t',y_t[2])
        # y_t TxSxNxAxS
        temp_x = np.zeros_like(x_t[0])
        # temp_y  SxNxAxS
        temp_y = np.zeros_like(y_t[0])
        for t in range(tau_l + 1, tau_l + T_l + 1):
            for state in range(S):
                temp_x[state] += t / S_l * x_t[t][state]
        # print(temp_x[0])
        # for state in range(S):
        #     print('sum',np.sum(temp_x[state]))
        for t in range(tau_l + 1, tau_l + T_l + 1):
            for state in range(S):
                for i in range(N):
                    temp_y[state][i] += t / S_l * y_t[t][state][i]
        # print("temp_x is ", temp_x)
        sum_temp_y = np.zeros((S, A, S), dtype=np.float64)
        for state in range(S):
            for i in range(N):
                sum_temp_y[state] += 1 / N * temp_y[state][i]
        # for state in range(S):
        #     for a in range(A):
        #         print('sum_temp_y',np.sum(sum_temp_y[state][a]))
        # print("sum_temp_y is ", sum_temp_y)
        # update v use temp_x,sum_temp_y

        for state in range(S):
            res = 0
            for a in range(A):
                res += -temp_x[state][a] * rewards[state][a][0]
                for s_prime in range(S):
                    # assert temp_x[state][a] >= 0
                    # assert sum_temp_y[state][a][s_prime] >= 0
                    # print(temp_x[state][a],sum_temp_y[state][a][s_prime])
                    # print(rewards[state][a][s_prime],lambd*v[epoch][s_prime])
                    res += temp_x[state][a] * (sum_temp_y[state][a][s_prime] * lambd * -1 * v[epoch][s_prime])
            v[epoch + 1][state] = -res
        # v = np.ones(S, dtype=np.float64)

        delta_F_v = np.max(np.abs(v[epoch + 1] - v[epoch]))
        if delta_F_v < delta:
            final_v = v[epoch + 1]
            print('converge')
            # print("v is ", v[epoch])
            # print("policy is", temp_x)
            break
        # else:
        # print("v is ", v[epoch])

    return final_v
