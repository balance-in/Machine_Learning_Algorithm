# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:25:22 2020

@author: balance
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import math
import sys
from matplotlib import colors
import scipy as sc
import os

def list_list(a,b):
    sum1 = 0
    for i,j in zip(a,b):
        sum1 += i * j
    return sum1
def list_num(num,martix):
    num1=[]
    for i in martix:
        num1.append(i*num)
    return num1
def list__list(a,b):
    num1=[]
    for i,j in zip(a,b):
        num1.append(i-j)
    return num1
def abs_list(a):
    num1 = 0
    for i in a:
        num1 += i * i
    return math.sqrt(num1)
d = 1
theta = 0.1
eta = 1
a = [0,0,0]
samples_w1=[[0.1,1.1,d],[6.8,7.1,d],[-3.5,-4.1,d],[2,2.7,d],[4.1,2.8,d],[3.1,5,d],[-0.8,-1.3,d],[0.9,1.2,d],[5,6.4,d],[3.9,4.0,d]]
samples_w2=[[7.1,4.2,d],[-1.4,-4.3,d],[4.5,0,d],[6.3,1.6,d],[4.2,1.9,d],[1.4,-3.2,d],[2.4,-4,d],[2.5,-6.1,d],[8.4,3.7,d],[4.1,-2.2,d]]
samples_w3=[[-3,-2.9,d],[0.5,8.7,d],[2.9,2.1,d],[-0.1,5.2,d],[-4,2.2,d],[-1.3,3.7,d],[-3.4,6.2,d],[-4.1,3.4,d],[-5.1,1.6,d],[1.9,5.1,d]]
samples_w4=[[-2,-8.4,d],[-8.9,0.2,d],[-4.2,-7.7,d],[-8.5,-3.2,d],[-6.7,-4,d],[-0.5,-9.2,d],[-5.3,-6.7,d],[-8.7,-6.4,d],[-7.1,-9.7,d],[-8,-6.3,d]]



na = 0.1
samples_all =[]
samples_all.extend(samples_w1[0:8])
samples_all.extend(samples_w2[0:8])
samples_all.extend(samples_w3[0:8])
samples_all.extend(samples_w4[0:8])                  

samples_test = []
samples_test.extend(samples_w1[8:10])
samples_test.extend(samples_w2[8:10])
samples_test.extend(samples_w3[8:10])
samples_test.extend(samples_w4[8:10])

samples_test = np.array(samples_test)
samples_all = np.array(samples_all)


Y = []

for i in range(0,4):
    for j in range(0,8):
        if i == 0:
            Y.append([1,0,0,0])
        if i == 1:
            Y.append([0,1,0,0])
        if i == 2:
            Y.append([0,0,1,0])
        if i == 3:
            Y.append([0,0,0,1])

Y = np.array(Y)
Y = Y.T

X = samples_all.T
X_T = X.T

S = np.linalg.inv(np.dot(X,X_T) + na * np.eye(3))
W = np.dot(np.dot(S,X),Y.T)


X_train = []
print("依次输入w1,w2,w3,w4的最后两个数据，得到下列分类结果")
for i in samples_test:
    rec = np.dot(W.T, i.T)
    print(np.argmax(rec)+1)
