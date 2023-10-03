# author: Yu Zhuodong
import timeit
from tqdm import trange

import numpy as np

from fast_algorithm.bisection_gamma.bisection_gamma_l1 import solve_gamma_bisection_l1
from fast_algorithm.bisection_gamma.bisection_gamma_l2 import solve_gamma_bisection_l2
from gurobi_bellman.gurobi_nominal_infty import solve_gurobi_linfty, solve_gurobi_infty
from gurobi_bellman.gurobi_nominal_l1 import solve_gurobi_l1
from gurobi_bellman.gurobi_nominal_l2 import solve_gurobi_l2
from fast_algorithm.infty_norm.infty_norm_sub import solve_infty_fast
from julien_FOM.Julien_infinity import first_order_method_linfty, first_order_method_linfty_VI
from julien_FOM.Julien_l1 import first_order_method_l1, first_order_method_l1_VI
from julien_FOM.Julien_l2 import first_order_method_l2, first_order_method_l2_VI
from mdp_Tool import build_b_sa


# ---------------------------compare bellman update speed---------------------------------#
def compare_bellman_operator_l1(sample_transitions, states, actions, rewards, Lambd, theta, N):
    F_v = np.zeros([len(states)])
    F_v_gurobi = np.zeros([len(states)])
    b = build_b_sa(F_v, rewards, states, actions, Lambd)

    state = np.random.randint(0, len(states))
    k = 10
    ou = np.zeros(k)
    gu = np.zeros(k)
    j_3 = np.zeros(k)
    for i in trange(k):
        # print('start')
        # select randome state

        start = timeit.default_timer()
        # pr = cProfile.Profile()
        # pr.enable()
        solve_gamma_bisection_l1(sample_transitions, state, actions, theta, b[state], N)
        # print('ou, bellman', solve_gamma_bisection_l1(sample_transitions, state, actions, theta, b[state], N))
        # value_iteration_ou(nStates)
        # pr.disable()
        # pr.print_stats(sort='cumtime')
        end = timeit.default_timer()
        ou[i] = end - start
        # print('ou', 'total+time', end - start)
        start1 = timeit.default_timer()
        solve_gurobi_l1(states, actions, rewards, sample_transitions, N, F_v_gurobi, theta, state=state)
        # print('gu, bellman', solve_gurobi_l1(states, actions, rewards, sample_transitions, N, F_v_gurobi, state=state))
        # value_iteration_gu(nStates)
        end1 = timeit.default_timer()
        gu[i] = end1 - start1
        # print('gu', 'total+time', end1 - start1)
        start = timeit.default_timer()
        first_order_method_l1(3, sample_transitions, states, actions, rewards, Lambd, theta, N)
        end = timeit.default_timer()
        j_3[i] = (end - start)
    print(np.array([np.mean(ou),np.mean(gu),np.mean(j_3)]))
    print('bellman', 'l1', 'our algorithm', ':', 'average time', np.mean(ou), 'std of time', np.std(ou))
    print('bellman', 'l1', 'gurobi algorithm', ':', 'average time', np.mean(gu), 'std of time', np.std(gu))
    print('bellman', 'l1', 'first order method 2 it', ':', 'average time', np.mean(j_3), 'std of time', np.std(j_3))


def compare_bellman_operator_l2(sample_transitions, states, actions, rewards, Lambd, theta, N):
    F_v = np.zeros([len(states)])
    F_v_gurobi = np.zeros([len(states)])
    b = build_b_sa(F_v, rewards, states, actions, Lambd)

    state = np.random.randint(0, len(states))
    k = 10
    ou = np.zeros(k)
    gu = np.zeros(k)
    j_3 = np.zeros(k)
    for i in trange(k):
        # print('start')
        # select randome state

        start = timeit.default_timer()
        # pr = cProfile.Profile()
        # pr.enable()
        res = solve_gamma_bisection_l2(sample_transitions, state, actions, theta, b[state], N)
        # print('ou, bellman', res)
        # 'ou_l2', solve_gamma_bisection_l2(sample_transitions, state, actions, theta, b[state], N))
        # value_iteration_ou(nStates)
        # pr.disable()
        # pr.print_stats(sort='cumtime')
        end = timeit.default_timer()
        ou[i] = end - start
        # print('ou', 'total+time', end - start)
        # start1 = timeit.default_timer()
        gu_value, gu_run_time = solve_gurobi_l2(states, actions, rewards, sample_transitions, N, F_v_gurobi, theta,
                                                state)
        # print('gu, bellman', gu_value)
        # print('gu_l2', solve_gurobi_l2(states, actions, rewards, sample_transitions, N, F_v_gurobi, state=state))
        # value_iteration_gu(nStates)
        # end1 = timeit.default_timer()
        gu[i] = gu_run_time
        # print('gu', 'total+time', end1 - start1)
        '''test first order method speed'''
        start = timeit.default_timer()
        first_order_method_l2(3, sample_transitions, states, actions, rewards, Lambd, theta, N)
        end = timeit.default_timer()
        j_3[i] = (end - start)
    print(np.array([np.mean(ou), np.mean(gu), np.mean(j_3)]))
    print('bellman', 'l2', 'our algorithm', ':', 'average time', np.mean(ou), 'std of time', np.std(ou))
    print('bellman', 'l2', 'gurobi algorithm', ':', 'average time', np.mean(gu), 'std of time', np.std(gu))
    print('bellman', 'l2', 'first order method 3 it', ':', 'average time', np.mean(j_3), 'std of time', np.std(j_3))
    return np.mean(ou), np.mean(gu), np.mean(j_3), np.std(ou), np.std(gu), np.std(j_3)


def compare_bellman_operator_linfty(sample_transitions, states, actions, rewards, Lambd, theta, N):
    F_v = np.zeros([len(states)])
    F_v_gurobi = np.zeros([len(states)])
    b = build_b_sa(F_v, rewards, states, actions, Lambd)

    state = np.random.randint(0, len(states))
    k = 10
    ou = np.zeros(k)
    gu = np.zeros(k)
    j_3 = np.zeros(k)
    for i in trange(k):
        # print('start')
        # select randome state

        start = timeit.default_timer()
        # pr = cProfile.Profile()
        # pr.enable()
        solve_infty_fast(sample_transitions, state, actions, theta, b[state], N)
        # print('our',solve_infty_fast(sample_transitions, state, actions, theta, b[state], N))
        # 'ou_l2', solve_gamma_bisection_l2(sample_transitions, state, actions, theta, b[state], N))
        # value_iteration_ou(nStates)
        # pr.disable()
        # pr.print_stats(sort='cumtime')
        end = timeit.default_timer()
        ou[i] = end - start
        # print('ou', 'total+time', end - start)
        # start1 = timeit.default_timer()

        # print('gu_l2', solve_gurobi_l2(states, actions, rewards, sample_transitions, N, F_v_gurobi, state=state))
        # value_iteration_gu(nStates)
        # end1 = timeit.default_timer()

        res, gu_run_time = solve_gurobi_infty(sample_transitions, state, states, actions, theta, b[state], N)
        # print('gurobi',res)
        gu[i] = gu_run_time
        # print('gu', 'total+time', end1 - start1)
        '''test first order method speed'''
        start = timeit.default_timer()
        first_order_method_linfty(3, sample_transitions, states, actions, rewards, Lambd, theta, N)
        end = timeit.default_timer()
        j_3[i] = (end - start)
    print(np.array([np.mean(ou),np.mean(gu),np.mean(j_3)]))
    print('bellman', 'linfty', 'our algorithm', ':', 'average time', np.mean(ou), 'std of time', np.std(ou))
    print('bellman', 'linfty', 'gurobi algorithm', ':', 'average time', np.mean(gu), 'std of time', np.std(gu))
    print('bellman', 'linfty', 'first order method 2 it', ':', 'average time', np.mean(j_3), 'std of time', np.std(j_3))


# ------------------use value iteration to verify the accuracy of the algorithm---------------------------------#


# --------------------------- value iteration  ---------------------------------#
def compare_VI_operator_l1(sample_transitions, states, actions, rewards, theta, N):
    start = timeit.default_timer()
    value_iteration_ou(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1, max_iter=100)
    end = timeit.default_timer()
    print('VI', 'l1', 'our algorithm', ':', 'total+time', end - start)

    start = timeit.default_timer()
    value_iteration_gu(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1, max_iter=100)
    end = timeit.default_timer()
    print('VI', 'l1', 'gurobi algorithm', ':', 'total+time', end - start)

    start = timeit.default_timer()
    value_iteration_l1_julian(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1,
                              max_iter=100)
    end = timeit.default_timer()
    print('VI', 'l1', 'julien_FOM algorithm', ':', 'total+time', end - start)


def compare_VI_operator_l2(sample_transitions, states, actions, rewards, theta, N):
    start = timeit.default_timer()
    value_iteration_l2_ou(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1, max_iter=100)
    end = timeit.default_timer()
    print('VI', 'l2', 'our algorithm', ':', 'total+time', end - start)

    start = timeit.default_timer()
    value_iteration_l2_gu(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1, max_iter=100)
    end = timeit.default_timer()
    print('VI', 'l2', 'gurobi algorithm', ':', 'total+time', end - start)

    start = timeit.default_timer()
    value_iteration_l2_julian(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1,
                              max_iter=100)
    end = timeit.default_timer()
    print('VI', 'l2', 'julien_FOM algorithm', ':', 'total+time', end - start)


def compare_VI_operator_linfty(sample_transitions, states, actions, rewards, theta, N):
    start = timeit.default_timer()
    value_iteration_linfty_ou(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1, max_iter=100)
    end = timeit.default_timer()
    print('VI', 'linfty', 'our algorithm', ':', 'total+time', end - start)

    start = timeit.default_timer()
    value_iteration_linfty_gu(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1, max_iter=100)
    end = timeit.default_timer()
    print('VI', 'linfty', 'gurobi algorithm', ':', 'total+time', end - start)

    start = timeit.default_timer()
    value_iteration_linfty_julian(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1,
                                  max_iter=50)
    end = timeit.default_timer()
    print('VI', 'linfty', 'julien_FOM algorithm', ':', 'total+time', end - start)


# --------------------------- value iteration for 1 iteration ---------------------------------#
def value_iteration_gu(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1, max_iter=100):
    # initialize V to be 0 for all states
    F_v_beofre = np.zeros(len(states))
    F_v_after = np.zeros(len(states))

    delta = 2 * lambd * eps / (1 - lambd)
    for i in range(max_iter):
        # print('F_v', ':', F_v_beofre, F_v_after)
        # b = build_b_sa(F_v_beofre, rewards, states, actions, lambd)
        # print(i, 'th', 'b', 'build')
        for state in range(len(states)):
            # print(i,'th','state',state,' b[state]',b[state].shape)
            # print(b[state])
            F_v_after[state], gu_run_time = solve_gurobi_l1(states, actions, rewards, sample_transitions, N, F_v_beofre,
                                                            theta, state)
        # print('the', i, 'iteration', 'F_v', ':', F_v_beofre,F_v_after)
        # update the state value and the maximum change
        # print(i, 'th', 'F_v', ':', F_v_after)
        delta_F_v = np.max(np.abs(F_v_after - F_v_beofre))
        # print('delta_F_v = ', delta_F_v, 'delta = ', delta)
        # check if convergence has been reached
        if delta_F_v < delta:
            print(F_v_after)
            # print('vi converged')
            break
        else:
            F_v_beofre = F_v_after.copy()
    # print('gu', 'VI', F_v_after)


def value_iteration_ou(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1, max_iter=100):
    """
    Value Iteration Algorithm.

    Arguments:
    transitions -- a dictionary where transitions[state][action] = list of (next_state, probability) tuples
    rewards -- a dictionary where rewards[state][action] = immediate reward
    discount -- discount factor
    theta -- threshold for stopping
    max_iter -- maximum number of iterations

    Returns:
    V -- a dictionary where V[state] = value of the state
    """

    # initialize V to be 0 for all states
    F_v_beofre = np.zeros(len(states))
    F_v_after = np.zeros(len(states))

    delta = 2 * lambd * eps / (1 - lambd)
    # print('stop delta', delta)
    for i in range(max_iter):
        # print('F_v', ':', F_v_beofre, F_v_after)
        b = build_b_sa(F_v_beofre, rewards, states, actions, lambd)
        # print(i, 'th', 'b', 'build')
        for state in range(len(states)):
            # print(i,'th','state',state,' b[state]',b[state].shape)
            # print(b[state])
            F_v_after[state] = solve_gamma_bisection_l1(sample_transitions, state, actions, theta, b[state], N)
        # print('the', i, 'iteration', 'F_v', ':', F_v_beofre,F_v_after)
        # update the state value and the maximum change
        # print(i, 'th', 'F_v', ':', F_v_after)
        delta_F_v = np.max(np.abs(F_v_after - F_v_beofre))
        # print('delta_F_v = ', delta_F_v, 'delta = ', delta)
        # check if convergence has been reached
        if delta_F_v < delta:
            print(F_v_after)
            # print('vi converged')
            break
        else:
            F_v_beofre = F_v_after.copy()
    # print('ou', 'VI', F_v_after)


def value_iteration_l1_julian(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1,
                              max_iter=100):
    """
    Value Iteration Algorithm.

    Arguments:
    transitions -- a dictionary where transitions[state][action] = list of (next_state, probability) tuples
    rewards -- a dictionary where rewards[state][action] = immediate reward
    discount -- discount factor
    theta -- threshold for stopping (default 0.00001)
    max_iter -- maximum number of iterations (default 10000)

    Returns:
    V -- a dictionary where V[state] = value of the state
    """

    # initialize V to be 0 for all states
    delta = 2 * lambd * eps / (1 - lambd)
    v = first_order_method_l1_VI(sample_transitions, states, actions, rewards, lambd, theta, N, delta, max_iter)
    print('julien_FOM', v)
    # print('ou', 'VI', 'l2', F_v_after)


def value_iteration_l2_gu(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1, max_iter=100):
    # initialize V to be 0 for all states
    F_v_beofre = np.zeros(len(states))
    F_v_after = np.zeros(len(states))

    delta = 2 * lambd * eps / (1 - lambd)
    # print('stop delta', delta)
    for i in range(max_iter):
        # print('F_v', ':', F_v_beofre, F_v_after)
        # b = build_b_sa(F_v_beofre, rewards, states, actions, lambd)
        # print(i, 'th', 'b', 'build')
        for state in range(len(states)):
            # print(i,'th','state',state,' b[state]',b[state].shape)
            # print(b[state])
            F_v_after[state], gu_run_time = solve_gurobi_l2(states, actions, rewards, sample_transitions, N, F_v_beofre,
                                                            theta, state)
        # print('the', i, 'iteration', 'F_v', ':', F_v_beofre,F_v_after)
        # update the state value and the maximum change
        # print(i, 'th', 'F_v', ':', F_v_after)
        delta_F_v = np.max(np.abs(F_v_after - F_v_beofre))
        # print('delta_F_v = ', delta_F_v, 'delta = ', delta)
        # check if convergence has been reached
        if delta_F_v < delta:
            # print('vi converged')
            print(F_v_after)
            break
        else:
            F_v_beofre = F_v_after.copy()


def value_iteration_l2_ou(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1, max_iter=100):
    """
    Value Iteration Algorithm.

    Arguments:
    transitions -- a dictionary where transitions[state][action] = list of (next_state, probability) tuples
    rewards -- a dictionary where rewards[state][action] = immediate reward
    discount -- discount factor
    theta -- threshold for stopping (default 0.00001)
    max_iter -- maximum number of iterations (default 10000)

    Returns:
    V -- a dictionary where V[state] = value of the state
    """

    # initialize V to be 0 for all states
    F_v_beofre = np.zeros(len(states))
    F_v_after = np.zeros(len(states))

    delta = 2 * lambd * eps / (1 - lambd)

    for i in range(max_iter):
        # print('F_v', ':', F_v_beofre, F_v_after)
        b = build_b_sa(F_v_beofre, rewards, states, actions, lambd)
        # print(i, 'th', 'b', 'build')
        for state in range(len(states)):
            # print(i,'th','state',state,' b[state]',b[state].shape)
            # print(b[state])
            F_v_after[state] = solve_gamma_bisection_l2(sample_transitions, state, actions, theta, b[state], N)
        # print('the', i, 'iteration', 'F_v', ':', F_v_beofre,F_v_after)
        # update the state value and the maximum change
        # print(i, 'th', 'F_v', ':', F_v_after)
        delta_F_v = np.max(np.abs(F_v_after - F_v_beofre))
        # print('delta_F_v = ', delta_F_v, 'delta = ', delta)
        # check if convergence has been reached
        if delta_F_v < delta:
            # print('vi converged')
            print(F_v_after)
            break
        else:
            F_v_beofre = F_v_after.copy()
            # print(F_v_beofre)


def value_iteration_l2_julian(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1,
                              max_iter=100):
    """
    Value Iteration Algorithm.

    Arguments:
    transitions -- a dictionary where transitions[state][action] = list of (next_state, probability) tuples
    rewards -- a dictionary where rewards[state][action] = immediate reward
    discount -- discount factor
    theta -- threshold for stopping (default 0.00001)
    max_iter -- maximum number of iterations (default 10000)

    Returns:
    V -- a dictionary where V[state] = value of the state
    """

    # initialize V to be 0 for all states
    delta = 2 * lambd * eps / (1 - lambd)
    v = first_order_method_l2_VI(sample_transitions, states, actions, rewards, lambd, theta, N, delta, max_iter)
    print('julien_FOM', v)
    # print('ou', 'VI', 'l2', F_v_after)


def value_iteration_linfty_gu(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1, max_iter=100):
    # initialize V to be 0 for all states
    F_v_beofre = np.zeros(len(states))
    F_v_after = np.zeros(len(states))

    delta = 2 * lambd * eps / (1 - lambd)
    # print('stop delta', delta)
    for i in range(max_iter):
        b_s = build_b_sa(F_v_beofre, rewards, states, actions, lambd)
        # print('F_v', ':', F_v_beofre, F_v_after)
        # b = build_b_sa(F_v_beofre, rewards, states, actions, lambd)
        # print(i, 'th', 'b', 'build')
        for state in range(len(states)):
            # print(i,'th','state',state,' b[state]',b[state].shape)
            # print(b[state])
            F_v_after[state], gu_run_time = solve_gurobi_linfty(sample_transitions, state, states, actions, theta,
                                                                b_s[state], N)
            # F_v_after[state], gu_run_time = solve_gurobi_linfty(sample_transitions, state, states, actions, theta, b_s,
            #                                                     N)
        # print('the', i, 'iteration', 'F_v', ':', F_v_beofre,F_v_after)
        # update the state value and the maximum change
        # print(i, 'th', 'F_v', ':', F_v_after)
        delta_F_v = np.max(np.abs(F_v_after - F_v_beofre))
        # print('delta_F_v = ', delta_F_v, 'delta = ', delta)
        # check if convergence has been reached
        if delta_F_v < delta:
            # print('vi converged')
            print('gu', F_v_after)
            break
        else:
            F_v_beofre = F_v_after.copy()
            # print(F_v_after)


def value_iteration_linfty_ou(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1, max_iter=100):
    """
    Value Iteration Algorithm.

    Arguments:
    transitions -- a dictionary where transitions[state][action] = list of (next_state, probability) tuples
    rewards -- a dictionary where rewards[state][action] = immediate reward
    discount -- discount factor
    theta -- threshold for stopping (default 0.00001)
    max_iter -- maximum number of iterations (default 10000)

    Returns:
    V -- a dictionary where V[state] = value of the state
    """

    # initialize V to be 0 for all states
    F_v_beofre = np.zeros(len(states))
    F_v_after = np.zeros(len(states))

    delta = 2 * lambd * eps / (1 - lambd)

    for i in range(max_iter):
        # print('F_v', ':', F_v_beofre, F_v_after)
        b = build_b_sa(F_v_beofre, rewards, states, actions, lambd)
        # print(i, 'th', 'b', 'build')
        for state in range(len(states)):
            # print(i,'th','state',state,' b[state]',b[state].shape)
            # print(b[state])
            F_v_after[state] = solve_infty_fast(sample_transitions, state, actions, theta, b[state], N)
        # print('the', i, 'iteration', 'F_v', ':', F_v_beofre,F_v_after)
        # update the state value and the maximum change
        # print(i, 'th', 'F_v', ':', F_v_after)
        delta_F_v = np.max(np.abs(F_v_after - F_v_beofre))
        # print('delta_F_v = ', delta_F_v, 'delta = ', delta)
        # check if convergence has been reached
        if delta_F_v < delta:
            # print('vi converged')
            print(F_v_after)
            break
        else:
            F_v_beofre = F_v_after.copy()
            # print(F_v_beofre)


def value_iteration_linfty_julian(sample_transitions, states, actions, rewards, theta, N, lambd=0.8, eps=0.1,
                                  max_iter=100):
    """
    Value Iteration Algorithm.

    Arguments:
    transitions -- a dictionary where transitions[state][action] = list of (next_state, probability) tuples
    rewards -- a dictionary where rewards[state][action] = immediate reward
    discount -- discount factor
    theta -- threshold for stopping (default 0.00001)
    max_iter -- maximum number of iterations (default 10000)

    Returns:
    V -- a dictionary where V[state] = value of the state
    """

    # initialize V to be 0 for all states
    delta = 2 * lambd * eps / (1 - lambd)
    v = first_order_method_linfty_VI(sample_transitions, states, actions, rewards, lambd, theta, N, delta, max_iter)
    print('julien_FOM', v)
    # print('ou', 'VI', 'l2', F_v_after)



