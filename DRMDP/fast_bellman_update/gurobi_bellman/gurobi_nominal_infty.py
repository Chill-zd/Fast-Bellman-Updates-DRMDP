# author: Yu Zhuodong
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum


def solve_gurobi_linfty(sample_transitions, state, states, actions, theta, b_s, N):
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

    x = {}

    # eps = {}
    for u in range(N):
        for k in range(len(states)):
            x[u, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'x_{u}_{k}')

    for u in range(N):
        model.addConstr(quicksum(x[u, k] for k in range(len(states))) == 1)

    for u in range(N):
        for k in range(len(states)):
            model.addConstr(x[u, k] - sample_transitions[u][state][action][k] <= theta)
            model.addConstr(sample_transitions[u][state][action][k] - x[u, k] <= theta)

    sum = gp.quicksum(x[u, k] * b_s[action][k] / N for u in range(N) for k in range(len(states)))

    model.setObjective(sum, sense=gp.GRB.MINIMIZE)
    model.optimize()

    return model.objVal, model.runtime


def solve_gurobi_infty(sample_transitions, state, states, actions, theta, b_s, N):
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

    return model.objVal, model.runtime
