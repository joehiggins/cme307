#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:49:21 2018

@author: lsmoore(not joe higgins)
"""
import numpy as np

A = np.matrix([
    [1, 3],
    [5, 2]
])

x_star = np.matrix([[3],[4]])
b = A*x_star

alpha = 0.25
beta = 0.65
meu = 0

def func(x):
    return 1/2 * np.linalg.norm(A*x - b)**2 - meu * np.sum(np.log(x))

def grad(x):
    return (np.transpose(A) * (A*x - b) - meu * 1./x)

def descent(x):
    return -1 * grad(x)


def new_t(x):
    t = 1
    while sum((x+t*descent(x)) < 0) > 0 or func(x + t*descent(x)) > func(x) + alpha * t * np.dot(np.transpose(descent(x)), grad(x)):
        t = t * beta
    return t

x0 = np.matrix([
    [1],
    [1]
])

x_k = x0
check = 9999
maxiter = 100000
k = 0
while(check > 10**-8 and k < maxiter):
    
    x_k1 = x_k + new_t(x_k) * descent(x_k)
    check = np.linalg.norm(x_k1 - x_k)
    x_k = x_k1
    k = k + 1

print(x_k)