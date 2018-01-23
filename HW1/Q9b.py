#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:49:18 2018

@author: josephhiggins
"""

import cvxpy as cvx
import numpy as np

def sum_elem_product(A,B):
    return cvx.sum_entries(cvx.mul_elemwise(A, B))

def col_vec_3elem(a,b,c):
    return np.matrix([[a],[b],[c]])

# Constraints will force Z to look like what its supposed to look like here
Z = Semidef(3)

sensor_location = [-0.35,0.45]

anchors = np.matrix([
    [ 1, 0],
    [-1, 0],
    [ 0, 2]
])

d = list(map(lambda a: np.linalg.norm(sensor_location - a), anchors))

objective = cvx.Minimize(cvx.sum_entries(Z))

v0 = col_vec_3elem(1,0,0)
v1 = col_vec_3elem(0,1,0)
v2 = col_vec_3elem(1,1,0)

a0 = col_vec_3elem(anchors[0,0],anchors[0,1],-1)
a1 = col_vec_3elem(anchors[1,0],anchors[1,1],-1)
a2 = col_vec_3elem(anchors[2,0],anchors[2,1],-1)

constraints = [
    sum_elem_product(v0*np.transpose(v0), Z) == 1,
    sum_elem_product(v1*np.transpose(v1), Z) == 1,
    sum_elem_product(v2*np.transpose(v2), Z) == 2,
    sum_elem_product(a0*np.transpose(a0), Z) == cvx.square(d[0]),
    sum_elem_product(a1*np.transpose(a1), Z) == cvx.square(d[1]),
    sum_elem_product(a2*np.transpose(a2), Z) == cvx.square(d[2])
]

prob = Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()
# The optimal value for x is stored in x.value.
print(Z[0,2].value)
print(Z[1,2].value)