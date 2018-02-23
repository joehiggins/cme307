#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:43:46 2018

@author: YinYu Ye
Is a cone
"""

import numpy as np
import random as rand
e = np.exp(1)

anchors = np.matrix([
    [1, 0],
    [-1, 0],
    [0, 2]
])

sensors = np.matrix([
    [-0.5, 0.5],
    [ 0.5, 0.5]
])

#define gradient function
def grad(X):
    sensor_distance_sum = np.zeros((1,2))
    anchor_distance_sum = np.zeros((1,2))
    gradient = np.zeros((2,2))

    for i, sensor_i in enumerate(X):
        for j, sensor_j in enumerate(X):
            if(i != j):
                sensor_distance_sum += (np.linalg.norm(sensor_i - sensor_j)**2 - \
                                        np.linalg.norm(sensors[i,:] - sensors[j,:])**2) * \
                                       (sensor_i - sensor_j)

        for k, anchor_k in enumerate(anchors):
            if(k - i >= 0 and k - i <= 1):
                anchor_distance_sum += (np.linalg.norm(anchor_k - sensor_i)**2 - \
                                        np.linalg.norm(anchors[k,:] - sensors[i,:])**2) * \
                                       (sensor_i - anchor_k)

        gradient[i,:] = 8*sensor_distance_sum + 4*anchor_distance_sum
        
    return gradient
        

#initialize looping variables
''' 
#SOCP result GUESS
sensors_0 = np.matrix([
    [-.440502, 0.14807],
    [-.40809, 0.74565],
])
'''
''' 
#SDP result GUESS
sensors_0 = np.matrix([
    [-.50000, 0.207108],
    [ .21649, 0.641756],
])
'''
sensors_0 = np.matrix([
    [-.50000, 0.207108],
    [ .21649, 0.641756],
])

check = 999
max_iter = 999999999
k = 0
sensors_k = sensors_0

#do iteration
alpha = .05
while check > 10**-8 and k < max_iter:

    print(sensors_k)
    sensors_k1 = sensors_k - alpha * grad(sensors_k)
    check = np.linalg.norm(sensors_k1 - sensors_k)
    sensors_k = sensors_k1
    k = k+1

sensors_k1
grad(sensors)