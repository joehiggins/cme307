#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 20:49:51 2018

@author: josephhiggins
"""

import cvxpy as cvx
import numpy as np

# Problem data.
n = 2 #number of dimensions

sensor_location = [0.31,0.15]

anchors = np.matrix([
    [ 1, 0],
    [-1, 0],
    [ 0, 2]
])

d = list(map(lambda a: np.linalg.norm(sensor_location - a), anchors))

# Construct the problem.
x = cvx.Variable(1,n)
objective = cvx.Minimize(cvx.norm(x - anchors[0]) + 
                         cvx.norm(x - anchors[1]) + 
                         cvx.norm(x - anchors[2]))

constraints = [cvx.norm(x - anchors[0]) <= d[0],
               cvx.norm(x - anchors[1]) <= d[1],
               cvx.norm(x - anchors[2]) <= d[2]]

prob = Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()
# The optimal value for x is stored in x.value.
print(x.value)
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
print(constraints[0].dual_value)