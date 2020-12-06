# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:48:06 2020

@author: balance
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:16:54 2020

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
import random



input_num = 3
mid_num =15
output_num = 3
X = np.zeros(3)
H = np.zeros(5)
Z = np.zeros(3)
eta = 0.5

x_train_1 = np.array([[ 1.58, 2.32, -5.8], [ 0.67, 1.58, -4.78], [ 1.04, 1.01, -3.63],
[-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
[ 1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [ 0.45, 1.33, -4.38],
[-0.76, 0.84, -1.96]])
x_train_2 = np.array([[ 0.21, 0.03, -2.21], [ 0.37, 0.28, -1.8], [ 0.18, 1.22, 0.16],
[-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
[-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [ 0.44, 1.31, -0.14],
[ 0.46, 1.49, 0.68]])
x_train_3 = np.array([[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [ 1.55, 0.99, 2.69],
[1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
[1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [ 0.25, 0.68, -0.99],
[ 0.66, -0.45, 0.08]])
t1 = np.array([1,0,0])
t2 = np.array([0,1,0])
t3 = np.array([0,0,1])
#np.random.uniform(0,1,2)  生成两个在区间[0,1]均匀分布采样数据，区间左闭右开
#np.random.randn(shape)生成shape数量的正太分布采样值


def sigmoid(inX):
    if inX>=0:
        return 1.0/(1+np.exp(-inX))
    else:
        return np.exp(inX)/(1+np.exp(inX))
def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))   
def tanh_derivative(x):
    return 1 - (np.tanh(x) * np.tanh(x))

def Forward_propagation(x,Wih,Whj):
    out_y = np.zeros(mid_num)
    out_z = np.zeros(output_num)
    net_y = np.zeros(mid_num)
    net_z = np.zeros(output_num)
    for j in range(0,len(Wih[0])):
        for i in range(0,len(Wih)):
            net_y[j] += Wih[i][j] *x[i]
        out_y[j] = np.tanh(net_y[j])
        
    for j in range(0,len(Whj[0])):
        for i in range(0,len(Whj)):
            net_z[j] += Whj[i][j] * out_y[i]
        out_z[j] = sigmoid(net_z[j])
    return out_y,net_y,out_z,net_z

def MSE(x,Wih,Whj,t):
    res = 0
    out_y,net_y,out_z,net_z = Forward_propagation(x,Wih,Whj)
    for i in range(0,output_num):
        res += (out_z[i] - t[i]) * (out_z[i] - t[i])
    return res / 2 
def Backward_propagation(X_train,t,Wih,Whj,del_Wih,del_Whj):
    out_y,net_y,out_z,net_z = Forward_propagation(X_train,Wih,Whj)
    error = t - out_z
    #计算theta_j
    theta_j = np.zeros(output_num)
    for i in range(0,output_num):
        theta_j[i] = sigmoid_derivative(net_z[i]) * error[i]
    #Whj
    for i in range(0,mid_num):
        for j in range(0,output_num):
            del_Whj[i][j] =  del_Whj[i][j] + eta * theta_j[j] * out_y[i]
    #theta_h
    theta_h = np.zeros(mid_num)
    for i in range(0,mid_num):
        res = 0
        for j in range(0,output_num):
            res += Whj[i][j] * theta_j[j]
        theta_h[i] = tanh_derivative(net_y[i]) * res
    #print(theta_h)
    #Wih
    for i in range(0,input_num):
        for j in range(0,mid_num):
            del_Wih[i][j] =  del_Wih[i][j] + eta * theta_h[j] * X_train[i]
    return del_Wih,del_Whj
def B_train(x_train_1,x_train_2,x_train_3,Wih,Whj):
    del_Wih = np.zeros((input_num,mid_num))
    del_Whj = np.zeros((mid_num,output_num))
    for x in x_train_1:
        del_Wih,del_Whj = Backward_propagation(x,t1,Wih,Whj,del_Wih,del_Whj)
    for x in x_train_2:
        del_Wih,del_Whj = Backward_propagation(x,t2,Wih,Whj,del_Wih,del_Whj)    
    for x in x_train_3:
        del_Wih,del_Whj = Backward_propagation(x,t3,Wih,Whj,del_Wih,del_Whj)    
    return Wih + del_Wih,Whj+del_Whj
# =============================================================================
# Wih = np.random.randn(input_num,mid_num)
# Whj = np.random.randn(mid_num,output_num)
# =============================================================================
Wih = np.random.uniform(-0.5/mid_num,0.5/mid_num,(input_num,mid_num))
Whj = np.random.uniform(-0.5/mid_num,0.5/mid_num,(mid_num,output_num))
np.save('Wih',Wih)
np.save('Whj',Whj)
# =============================================================================
# Wih = np.load('Wih.npy')
# Whj = np.load('Whj.npy')
# =============================================================================
count = 0
while(1):
    count = count + 1
    if (count > 1000):
        break
    Wih,Whj = B_train(x_train_1,x_train_2,x_train_3,Wih,Whj)
# =============================================================================
#     Wih,Whj = Backward_propagation(x_train_2,t2,Wih,Whj)
#     Wih,Whj = Backward_propagation(x_train_3,t3,Wih,Whj)
# =============================================================================

jw_list = []
out_y,net_y,out_z,net_z = Forward_propagation(x_train_1[1],Wih,Whj)
print(out_z)
out_y,net_y,out_z,net_z = Forward_propagation(x_train_2[1],Wih,Whj)
print(out_z)
out_y,net_y,out_z,net_z = Forward_propagation(x_train_3[1],Wih,Whj)
print(out_z)

for i in range(0,10):
    jw_list.append(MSE(x_train_1[i],Wih,Whj,t1))
    jw_list.append(MSE(x_train_2[i],Wih,Whj,t2))
    jw_list.append(MSE(x_train_3[i],Wih,Whj,t3))
print("单隐层节点个数：" + str(mid_num))
print("平均误差：" + str(sum(jw_list) / len(jw_list)))
# =============================================================================
# print("隐含层-输出层的权重矩阵：\n")
# print(Whj)
# print("输入层-隐含层的权重矩阵：\n")
# print(Wih)    
# =============================================================================
