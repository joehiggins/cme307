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

b = np.matrix([[-2],[6]])

alpha = 0.25

beta = 0.5

meu = 0.01

def func(x):
    return 1/2 * np.linalg.norm(A*x - b)**2 - meu * np.sum(np.log(x))

def grad(x):
    return (np.transpose(A) * (A*x - b) - meu * 1./x)

def descent(x):
    return -1 * grad(x)


def new_t(x):
    t = 1
    while func(x + t*descent(x)) > func(x) + alpha * np.dot(np.transpose(descent(x)), grad(x)):
        t = t * beta
    return t

x0 = np.matrix([[1],[1]])

x_k = x0
check = 9999
maxiter = 100000
k = 0
while(check > 10**-8 and k < maxiter):
    
    x_k1 = x_k + new_t(x_k) * descent(x_k)
    check = np.linalg.norm(x_k1 - x_k)
    print(check)
    x_k = x_k1
    k = k + 1
x_k

x = x0