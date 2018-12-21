# import cvxpy
# import cvxopt
# from cvxopt.modeling import variable, max
import scipy
import scipy.optimize

import numpy as np


# # cvxopt
# var_0 = cvxopt.modeling.variable(1, "var_0")
# var_1 = cvxopt.modeling.variable(1, "var_1")
# var_2 = cvxopt.modeling.variable(1, "var_2")
# max_diff = cvxopt.modeling.variable(1, "max_diff")
#
# constraints = []
#
# constraints.append((max_diff == cvxopt.modeling.max(var_0, var_1, var_2)))
#
# constraints.append((var_0 >= 0))
# constraints.append((var_1 >= 0))
# constraints.append((var_2 >= 0))
# constraints.append((max_diff >= 0))
#
# constraints.append((var_0 + var_1 + var_2 == 1))
#
# constraints.append((max_diff == var_0 + var_1 + var_2))
#
# problem = cvxopt.modeling.op(max_diff,
#                              constraints)
#
# problem.solve()
# print(problem.status)

#
# cvxpy
# var_0 = cvxpy.Variable()
# var_1 = cvxpy.Variable()
# var_2 = cvxpy.Variable()
#
# max_diff = cvxpy.Variable()
#
# constraints = [var_0 >= 0, var_1 >= 0, var_2 >= 0,
#                var_0 + var_1 + var_2 <= 1,
#                max_diff == cvxpy.maximum(var_0, var_1, var_2)]  # - cvxpy.minimum(var_0, var_1, var_2)) ** 2]
#
# objective = cvxpy.Maximize(max_diff)
#
# prob = cvxpy.Problem(objective, constraints)
#
# result = prob.solve(verbose=True)

def test_sum(x):
    return np.sum(x)


def test_square(x):
    return np.sum(np.square(x))


def max_diff(x):
    return np.max(x) - np.min(x)


def min_func(x):
    return x[0]


constr = []

#constr.append({'type': 'eq', 'fun': lambda x: test_square(x) -1 })
constr.append({'type': 'eq', 'fun': lambda x: test_sum(x) })
constr.append({'type': 'eq', 'fun': lambda x: max_diff(x) - 0.3})

np.random.seed(123)
g = np.random.rand(5, 5)

g = scipy.optimize.minimize(fun=min_func, x0=g, constraints=constr)
assert g.success

result = g.x

i = 1
