import numpy as np
import math

from scipy.optimize import linear_sum_assignment

''' ======================================================================================================= '''
    ###                                        OR Tools                                                 ###
''' ======================================================================================================= '''


def degenerate_assignment(C: np.array, feasible: np.array) -> np.array:
    from ortools.linear_solver import pywraplp

    n1,n2 = C.shape
    assert(n2 <= n1)

    degeneracy = int( math.ceil( (n1 / n2)**2 ) )

    rows,cols = np.argwhere(feasible).T
    solver    =  pywraplp.Solver.CreateSolver('SCIP') ## play with this

    x = {}
    for i,j in zip(rows,cols):
        x[i,j] = solver.IntVar(0, 1, '')

    for i in range(n1):
        constr = solver.Sum([x[i,j] for j in cols[rows==i]])
        solver.Add(1 == constr)

    for j in range(n2):
        constr = solver.Sum([x[i,j] for i in rows[cols==j]])
        solver.Add(1 <= constr)
        solver.Add(constr <= degeneracy)

    solver.Maximize(solver.Sum([C[i,j]*x[i,j] for i,j in zip(rows,cols)]))

    info = "Number of variables = {0}, number of constraints = {1}".format(solver.NumVariables(), solver.NumConstraints())
    #print(info)
    status = solver.Solve()
    X = np.zeros((n1,n2), dtype = np.int)

    if (status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE):
        for i,j in zip(rows,cols):
            X[i,j] = x[i,j].solution_value()

    a,b = np.argwhere(X).T
    return a,b


''' ======================================================================================================= '''
    ###                                           End                                                   ###
''' ======================================================================================================= '''


if __name__ == '__main__':
    pass
