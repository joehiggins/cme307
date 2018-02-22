#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:43:46 2018
@author: lawrencemoore
"""

import numpy as np
import random as rand
e = np.exp(1)

#define gradient function
def grad(x):
    
    pos_data_sum = \
        e**(-1*np.dot(pos_data[0,:], x))/(1 + e**(-1*np.dot(pos_data[0,:], x))) * pos_data[0,:] + \
        e**(-1*np.dot(pos_data[1,:], x))/(1 + e**(-1*np.dot(pos_data[1,:], x))) * pos_data[1,:] + \
        e**(-1*np.dot(pos_data[2,:], x))/(1 + e**(-1*np.dot(pos_data[2,:], x))) * pos_data[2,:]

    neg_data_sum = \
        e**(np.dot(neg_data[0,:], x))/(1 + e**(np.dot(neg_data[0,:], x))) * neg_data[0,:] + \
        e**(np.dot(neg_data[1,:], x))/(1 + e**(np.dot(neg_data[1,:], x))) * neg_data[1,:] + \
        e**(np.dot(neg_data[2,:], x))/(1 + e**(np.dot(neg_data[2,:], x))) * neg_data[2,:]
        
    return np.transpose(-1*pos_data_sum + neg_data_sum)

def new_alpha(x_k, x_km1):
    delta_x = np.subtract(x_k, x_km1)
    delta_grad_x = np.subtract(grad(x_k), grad(x_km1))
    return np.asscalar(np.dot(np.transpose(delta_x), delta_grad_x) / np.dot(np.transpose(delta_grad_x), delta_grad_x))

#add positive and negative data
pos_data = np.matrix([
    [ 0, 0],
    [ 1, 0],
    [ 0, 1]
])
neg_data = np.matrix([
    [ 0, 0],
    [-1, 0],
    [ 0,-1]
])

#add noise
pos_data = pos_data + np.matrix((np.random.rand(3,2)*0.001))
neg_data = neg_data + np.matrix((np.random.rand(3,2)*0.001))

#add shift variable
pos_data = np.hstack((pos_data, np.ones((3,1))))
neg_data = np.hstack((neg_data, np.ones((3,1))))
    
#initialize looping variables
x_0 = np.matrix([
    [-1], 
    [1],
    [0]
])
check = 999
max_iter = 1000
k = 0
x_km1 = x_0


alpha = 0.5
x_k = x_km1 - alpha * grad(x_km1)
check = np.linalg.norm(x_k - x_km1)


#do iteration
while check > 10**-8 and k < max_iter:
    alpha = new_alpha(x_k, x_km1)
    x_kp1 = x_k - alpha * grad(x_k)
    check = np.linalg.norm(x_kp1 - x_k)

    print(check)
    x_km1 = x_k
    x_k = x_kp1
    k = k+1

print(x_km1)
