# author: Yu Zhuodong
import numpy as np
import numba as nb


@nb.njit
def project(r, a, b):
    n = r.size
    total = np.sum(a)
    lambdas = np.append(a - r, b - r)
    idx = np.argsort(lambdas)
    lambdas = lambdas[idx]
    active = 1
    for i in range(1, 2 * n):
        total += active * (lambdas[i] - lambdas[i - 1])
        if total >= 1:
            lam = (1 - total) / active + lambdas[i]
            return np.clip(r + lam, a, b)
        elif idx[i] < n:
            active += 1
        else:
            active -= 1


@nb.njit
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
def cal_linfty(y_prime, y_hat, sigma, h, theta, N, A, state, y_t, t):
    res = 0
    for i in range(N):
        for a in range(A):
            temp_y_s_t = y_prime[i][a]
            temp_y_hat = y_hat[i][a]

            lb = np.clip(temp_y_hat - theta, 0, None)
            ub = np.clip(temp_y_hat + theta, None, 1)
            temp = temp_y_s_t - sigma * h[a]
            optimal_y = project(temp, lb, ub)
            y_t[t + 1][state][i][a] = optimal_y
    # res = res / N
    return y_t[t + 1][state]


@nb.njit
def first_order_method_linfty(epochs, sample_transition, states, actions, rewards, lambd, theta, N):
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

                y_t[t + 1][state] = cal_linfty(y_t[t][state],
                                               sample_transition[:, state, :, :], sigma, h, theta, N,
                                               A,
                                               state, y_t, t)
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

    return 0


def first_order_method_linfty_VI(sample_transition, states, actions, rewards, lambd, theta, N, delta, max_iter):
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

                y_t[t + 1][state] = cal_linfty(y_t[t][state],
                                               sample_transition[:, state, :, :], sigma, h, theta, N,
                                               A,
                                               state, y_t, t)

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
    # print('done')
    # print('done')
