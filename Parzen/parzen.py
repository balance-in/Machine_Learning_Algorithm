# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:07:10 2020

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


h = 0.1

hn = h / math.sqrt(50)
A = 1/math.sqrt(2 * math.pi)
samples_all = [4.6019,5.2564,5.2200,3.2886,3.7942,
3.2271,4.9275,3.2789,5.7019,3.9945,
3.8936,6.7906,7.1624,4.1807,4.9630,
6.9630,4.4597,6.7175,5.8198,5.0555,
4.6469,6.6931,5.7111,4.3672,5.3927,
4.1220,5.1489,6.5319,5.5318,4.2403,
5.3480,4.3022,7.0193,3.2063,4.3405,
5.7715,4.1797,5.0179,5.6545,6.2577,
4.0729,4.8301,4.5283,4.8858,5.3695,
4.3814,5.8001,5.4267,4.5277,5.2760]

def Gauss_fun(x):
    B = A * pow(math.e, -pow(x,2) / 2)
    return B
def Aquare_parzen(x,x_all):
    sum1 = 0
    for i in x_all:
        if abs(x - i) <= hn / 2:
            sum1 +=1
    return sum1
def Gauss_parzen(x,x_all):
    sum1 = 0
    for i in x_all:
        sum1 += Gauss_fun((x - i)/hn) # stats.norm.pdf(((x - i)/hn))
    return sum1


x = np.arange(int(min(samples_all)) - 1, int(max(samples_all)) + 1,0.01)
y1_Aquare = []
y2_Gauss = []
for i in x:
    y1_Aquare.append(Aquare_parzen(i,samples_all))
    y2_Gauss.append(Gauss_parzen(i,samples_all))

y3 = np.zeros(50)
y1_Aquare = np.array([i / 50 / hn for i in y1_Aquare])
y2_Gauss = np.array([i / 50 / hn for i in y2_Gauss])
plt.figure()    
plt.plot(x, y1_Aquare,label='Rectangle_parzen')
plt.plot(x, y2_Gauss,label='Gauss_parzen')
plt.scatter(np.array(samples_all), y3,label='samples',s = 5, c = 'r',marker = '*')

plt.legend()
#plt.xlabel('Distance(km)')
#plt.ylabel('Min_Zh(dBz)')
plt.title('h = ' + str(h))
plt.grid(ls='--')
if h < 1:
    plt.savefig('Parzen h = ' +'01',dpi = 300)
else:  
    plt.savefig('Parzen h = ' + str(h),dpi = 300)


