#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 14:32:23 2018

@author: josephhiggins
"""


import numpy as np
import scipy as sp

sensors= np.matrix([
    [0, 0.1],
    [0, 0.4]
])

anchors = np.matrix([
    [ 1, 0],
    [-1, 0],
    [ 0, 2]
])

A_1 = np.matrix([[1], [0], [0], [0]]) * np.transpose(np.matrix([[1], [0], [0], [0]]))
A_2 = np.matrix([[0], [1], [0], [0]]) * np.transpose(np.matrix([[0], [1], [0], [0]]))
A_3 = np.matrix([[1], [1], [0], [0]]) * np.transpose(np.matrix([[1], [1], [0], [0]]))

A_4 = np.vstack((np.transpose(anchors[0]), -1, 0)) * np.transpose(np.vstack((np.transpose(anchors[0]), -1, 0)))
A_5 = np.vstack((np.transpose(anchors[1]), -1, 0)) * np.transpose(np.vstack((np.transpose(anchors[1]), -1, 0)))

A_6 = np.vstack((np.transpose(anchors[1]), 0, -1)) * np.transpose(np.vstack((np.transpose(anchors[1]), 0, -1)))
A_7 = np.vstack((np.transpose(anchors[2]), 0, -1)) * np.transpose(np.vstack((np.transpose(anchors[2]), 0, -1)))

A_8 = np.matrix([[0], [0], [1], [-1]]) * np.transpose(np.matrix([[0], [0], [1], [-1]]))


A = [A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8]

b = np.matrix([
    [1],
    [1],
    [2],
    
    [np.linalg.norm(anchors[0] - sensors[0])**2],
    [np.linalg.norm(anchors[1] - sensors[0])**2],
    
    [np.linalg.norm(anchors[1] - sensors[1])**2],
    [np.linalg.norm(anchors[2] - sensors[1])**2],
    
    [np.linalg.norm(sensors[0] - sensors[1])**2]
])

def sum_mult(A, X):
    return np.sum(np.multiply(A, X))
    
def transform(X):
    V = np.matrix([
        [sum_mult(A_1, X)],
        [sum_mult(A_2, X)],
        [sum_mult(A_3, X)],
        [sum_mult(A_4, X)],
        [sum_mult(A_5, X)],
        [sum_mult(A_6, X)],
        [sum_mult(A_7, X)],
        [sum_mult(A_8, X)],
    ])
    return V

def func(X):
    return 1/2 * np.linalg.norm(transform(X) - b)**2

def phi(X):
    return func(X) - mu * np.log(np.linalg.det(X))
    
def grad(X):
    vec = transform(X) - b
    cumsum = 0
    for i, A_i in enumerate(A):
        cumsum = cumsum + np.multiply(A_i, vec[i])
    
    X_sqrt = sp.linalg.sqrtm(X)
    
    return np.dot(np.dot(X_sqrt, cumsum), X_sqrt) - (mu * np.identity(X.shape[0]))

sensors_0 = np.matrix([
    [ .1, .1],
    [ 0, .4]
])
    
Y = np.dot(np.transpose(sensors_0), sensors_0)

X0 = np.matrix([
    [             1,              0, sensors_0[0,0], sensors_0[0,1]],
    [             0,              1, sensors_0[1,0], sensors_0[1,1]],
    [sensors_0[0,0], sensors_0[1,0],         Y[0,0],         Y[0,1]],
    [sensors_0[0,1], sensors_0[1,1],         Y[1,0],         Y[1,1]],
])

X_k = X0
check = 9999
maxiter = 999999
k = 0

X = X0

mu = 0.01
alpha = 0.01
while(check > 10**-8 and k < maxiter):
    
    print(np.real(X_k))
    
    X_k1 = X_k - alpha * grad(X_k)
    check = np.linalg.norm(X_k1 - X_k)
    X_k = X_k1
    k = k + 1
    
np.real(X_k1)
#X_k1[2:4,0:2]
#grad(X0)
    