# author: Yu Zhuodong
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum


def solve_gurobi_linfty(sample_transitions, state, states, actions, theta, b_s, N):
    '''
    using gurobi to solve sub problem, verify the correctness of value iteration
    :return: V(state), gurobi_runtime
    '''
    sum = np.zeros(len(actions))
    count_time = 0
    for action in range(len(actions)):
        sum[action], temp_time = solve_gurobi_linfty_sub(sample_transitions, state, states, action, actions, theta, b_s,
                                                         N)
        count_time += temp_time
    start = time.time()
    res = np.max(sum)
    end = time.time()
    count_time = count_time + end - start

    return res, count_time


def solve_gurobi_linfty_sub(sample_transitions, state, states, action, actions, theta, b_s, N):
    lambd = 0.8
    # build varaibles
    env = gp.Env(empty=True)
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.setParam(GRB.Param.Threads, 1)
    model.setParam("Presolve", 0)
    # model.params.NonConvex = 2

    x = {}

    # eps = {}
    for u in range(N):
        for k in range(len(states)):
            x[u, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'x_{u}_{k}')

    # model.update()
    # build constraints

    # add constaints : \sum_{k} x_{k} = 1
    for u in range(N):
        model.addConstr(quicksum(x[u, k] for k in range(len(states))) == 1)

    # add constarints: abs(x - x_hat) <= theta
    for u in range(N):
        for k in range(len(states)):
            model.addConstr(x[u, k] - sample_transitions[u][state][action][k] <= theta)
            model.addConstr(sample_transitions[u][state][action][k] - x[u, k] <= theta)

    # for u in range(len(sample_transitions)):
    #    for j in range(len(actions)):
    #        for k in range(len(states)):
    #            model.addConstr((x[u, j, k] - sample_transitions[u, state, j, k])*(x[u, j, k] - sample_transitions[u, state, j, k]) <= eps[u, j, k])

    '''
    model.addConstrs(
        (gp.quicksum(abs(eps[u, j, k]) for k in range(len(states))) <= alpha[u, j] for u in range(len(sample_transitions)) for j in
         range(len(actions))), name="constraint")
    '''
    # sum = {}
    sum = gp.quicksum(x[u, k] * b_s[action][k] / N for u in range(N) for k in range(len(states)))
    # for u in range(N):
    #     for k in range(len(states)):
    #         sum += x[u, k] * b_s[action][k]
    # sum /= N
    model.setObjective(sum, sense=gp.GRB.MINIMIZE)
    model.optimize()
    '''
    if model.status == GRB.OPTIMAL:
        print(f'Optimal solution found with objective value {model.objVal:.4f}')
    else:
        print('No optimal solution found.')

    '''
    '''print the solution

    all_vars = model.getVars()
    values = model.getAttr("X", all_vars)
    names = model.getAttr("VarName", all_vars)

    for name, val in zip(names, values):
        print(f"{name} = {val}")

    print('objVal', model.objVal)
    '''

    return model.objVal, model.runtime


def solve_gurobi_infty(sample_transitions, state, states, actions, theta, b_s, N):
    '''
    using gurobi to solve problem with infinity norm
    :return:
    '''
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.setParam(GRB.Param.Threads, 1)
    model.setParam("Presolve", 0)

    lambd = {}
    alpha = {}
    beta = {}
    gamma = {}
    pi = {}
    for a in range(len(actions)):
        pi[a] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'pi_{a}')

    for i in range(N):
        for a in range(len(actions)):
            lambd[i, a] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'lambda_{i}_{a}')
            for s in range(len(states)):
                alpha[i, a, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'alpha_{i}_{a}_{s}')
                beta[i, a, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'beta_{i}_{a}_{s}')
                gamma[i, a, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'gamma_{i}_{a}_{s}')

    model.update()

    # pi \in \delta_a
    model.addConstr(gp.quicksum(pi[a] for a in range(len(actions))) == 1)

    for i in range(N):
        for a in range(len(actions)):
            for s in range(len(states)):
                model.addConstr(pi[a] * b_s[a][s] + alpha[i, a, s] - beta[i, a, s] - gamma[i, a, s] + lambd[i, a] == 0)

    obj = - gp.quicksum(lambd[i, a] for i in range(N) for a in range(len(actions))) \
          + gp.quicksum((alpha[i, a, s] * (-sample_transitions[i][state][a][s] - theta) + beta[i, a, s] * (
            sample_transitions[i][state][a][s] - theta)) for i in range(N) for a in range(len(actions)) for s in
                        range(len(states)))
    model.setObjective(obj / N, sense=gp.GRB.MAXIMIZE)

    model.optimize()

    # all_vars = model.getVars()
    # values = model.getAttr("X", all_vars)
    # names = model.getAttr("VarName", all_vars)
    #
    # for name, val in zip(names, values):
    #     print(f"{name} = {val}")
    return model.objVal, model.runtime
