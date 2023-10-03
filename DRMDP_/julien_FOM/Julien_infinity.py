# author: Yu Zhuodong
import numpy as np
import numba as nb


@nb.njit
def project(r, a, b):
    '''
    solve l2 projection problem with box constraints
    :param r:
    :param a:
    :param b:
    :return:
    '''
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
    # print the shape of all inputs of the function:
    # print(y_prime.shape, y_hat.shape,  h.shape,y_t.shape)

    # print('sigma, gamma, y_prime, y_hat, h, theta, N',sigma, gamma, y_prime, y_hat, h, theta, N)
    res = 0
    for i in range(N):
        for a in range(A):
            temp_y_s_t = y_prime[i][a]
            temp_y_hat = y_hat[i][a]

            lb = np.clip(temp_y_hat - theta, 0, None)
            ub = np.clip(temp_y_hat + theta, None, 1)
            temp = temp_y_s_t - sigma * h[a]
            optimal_y = project(temp, lb, ub)
            # res += (np.dot(optimal_y - temp, optimal_y - temp) + np.dot(temp_y_s_t, temp_y_s_t) - np.dot(temp,
            #                                                                                              temp)) / (
            #                2 * sigma)
            y_t[t + 1][state][i][a] = optimal_y
    # res = res / N
    return y_t[t + 1][state]


@nb.njit
def first_order_method_linfty(epochs, sample_transition, states, actions, rewards, lambd, theta, N):

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

                y_t[t + 1][state] = cal_linfty(y_t[t][state],
                                               sample_transition[:, state, :, :], sigma, h, theta, N,
                                               A,
                                               state, y_t, t)
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


def first_order_method_linfty_VI(sample_transition, states, actions, rewards, lambd, theta, N, delta, max_iter):
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

                y_t[t + 1][state] = cal_linfty(y_t[t][state],
                                               sample_transition[:, state, :, :], sigma, h, theta, N,
                                               A,
                                               state, y_t, t)
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
            print("v is ", v[epoch + 1])
            # print("policy is", temp_x)
            break
        else:
            print("v is ", v[epoch + 1])

    return final_v
    # print('done')
    # print('done')
