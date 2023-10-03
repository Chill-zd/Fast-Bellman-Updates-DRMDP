# author: Yu Zhuodong
import gurobipy as gp
from gurobipy import GRB, quicksum

'''
initialization of nominal MDP and the sample MDPs
'''


def solve_gurobi_l1(states, actions, rewards, sample_transitions, N, F_v_gurobi, theta, state):
    rewards = rewards
    sample_transitions = sample_transitions

    lambd = 0.8
    # build varaibles
    env = gp.Env(empty=True)
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.setParam(GRB.Param.Threads, 1)
    model.setParam("Presolve", 0)
    #model.setParam('TimeLimit', 600)  # set time limit as 10 minutes

    x = {}
    alpha = {}
    eps = {}
    for u in range(len(sample_transitions)):
        for j in range(len(actions)):
            alpha[u, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY,
                                       name=f'alpha_{u}_{j}')
            for k in range(len(states)):
                x[u, j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'x_{u}_{j}_{k}')

                eps[u, j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY,
                                            name=f'eps_{u}_{j}_{k}')
    gamma = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='gamma')

    # model.update()
    # build constraints
    for j in range(len(actions)):  # for each action : 1/N \sum p*(r+\lambda*F_v) <= gamma
        model.addConstr(quicksum(
            x[u, j, k] * (rewards[state, j, k] + lambd * F_v_gurobi[k]) for k in range(len(states)) for u in
            range(len(sample_transitions))) <= len(sample_transitions) * gamma)
    model.addConstr(
        quicksum(alpha[u, j] for u in range(len(sample_transitions)) for j in range(len(actions))) <= len(
            sample_transitions) * theta)

    # add constaints : \sum_{k} x_{u,j,k} = 1
    for u in range(len(sample_transitions)):
        for j in range(len(actions)):
            model.addConstr(quicksum(x[u, j, k] for k in range(len(states))) == 1)

    for u in range(len(sample_transitions)):
        for j in range(len(actions)):
            model.addConstr(quicksum(eps[u, j, k] for k in range(len(states))) <= alpha[u, j])

    for u in range(len(sample_transitions)):
        for j in range(len(actions)):
            for k in range(len(states)):
                model.addConstr(x[u, j, k] - sample_transitions[u, state, j, k] <= eps[u, j, k])
                model.addConstr(x[u, j, k] - sample_transitions[u, state, j, k] >= -eps[u, j, k])
    '''
    model.addConstrs(
        (gp.quicksum(abs(eps[u, j, k]) for k in range(len(states))) <= alpha[u, j] for u in range(len(sample_transitions)) for j in
         range(len(actions))), name="constraint")
    '''

    model.setObjective(gamma, sense=gp.GRB.MINIMIZE)
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
    ###########-------------------- print the runtime ---------------------------###########
    # print('runtime', model.runtime)

    return model.objVal, model.runtime
