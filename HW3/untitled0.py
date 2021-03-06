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

b = np.matrix([[-2, 6]])

meu = 0.01

def func(x):
    return 1/2 * np.linalg.norm(A*x - b)**2 - meu * np.log(x)

def grad(x):
    return (np.tranpose(A) * (A*x - b) - meu * 1./x)

def descent(x):
    return -1 * grad(x)


def new_t(x, alpha, beta):
    t = 1
    while func(x + t*descent(x)) > func(x) + alpha * np.dot(grad(x), descent(x)):
        t = t * beta
    return t