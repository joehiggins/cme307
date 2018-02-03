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

anchors = np.matrix([
    [ 1, 0],
    [-1, 0],
    [ 0, 2]
])

sensors = np.matrix([
    [ 0, .1],
    [0, 0.5]        
])

d_sa = list(map(lambda s: 
        list(map(lambda a: np.linalg.norm(a - s), anchors))
    , sensors))

d_ss = list(map(lambda s1: 
        list(map(lambda s2: np.linalg.norm(s1 - s2), sensors))
    , sensors))

# Construct the problem.
x = cvx.Variable(2,n)
objective = cvx.Minimize(0)

constraints = [
    cvx.norm(x[0,:] - anchors[0]) <= d_sa[0][0],
    cvx.norm(x[0,:] - anchors[1]) <= d_sa[0][1],
    cvx.norm(x[1,:] - anchors[1]) <= d_sa[1][1],
    cvx.norm(x[1,:] - anchors[2]) <= d_sa[1][2],
    cvx.norm(x[0,:] - x[1,:]) <= d_ss[0][1]
]

prob = cvx.Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve(solver = 'MOSEK')
# The optimal value for x is stored in x.value.
print(x.value)

# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
#print(constraints[0].dual_value)