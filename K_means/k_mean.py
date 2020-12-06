import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.pylab import mpl


class K_means():
    def __init__(self,Data,k=4):
        self.Data = Data
        self.Data_num = len(Data)
        self.Data_eigenvalue_num = len(Data[0])
        self.k = k
        self.k_initialize = np.array([self.Data[random.randint(0, self.Data_num-1)] for i in range(0, self.k)])
        self.k_class = [[] for i in range(0, self.k)]
    def Get_k_class(self):
        return self.k_class
    def Get_k_initialize(self):
        return self.k_initialize
    def Distace(self,x,y):
        return np.linalg.norm((x-y),ord=2)
    def U_mean(self,x):
        if (len(x) == 0):
            return np.zeros((1, self.Data_eigenvalue_num))
        count = np.zeros((1, self.Data_eigenvalue_num))
        for i in range(0, len(x)):
            count += x[i]
        return count / len(x)
    def J_square(self,x):
        sum_num = 0
        for i in x:
            sum_num += i * i
        return sum_num
    def MSE(self):
        sum_num = 0
        for i, j in zip(self.k_class, self.k_initialize):
            for k in i:
                sum_num += self.J_square(np.array(k) - j)
        return sum_num
    def Train_k_means(self):
        count = 0
        while (1):
            rec = 0
            count = count + 1
            for i in dataSet:
                num_min = 999
                num_key = 0
                for j in range(0, len(self.k_initialize)):
                    if self.Distace(np.array(i), self.k_initialize[j]) < num_min:
                        num_min = self.Distace(i, self.k_initialize[j])
                        num_key = j
                self.k_class[num_key].append(i)
            for i in range(0, self.k):
                new_u = self.U_mean(np.array(self.k_class[i]))
                if (new_u == self.k_initialize[i]).all():
                    rec += 1
                else:
                    self.k_initialize[i] = new_u
            if (rec == 4):
                break
            else:
                self.k_class = [[] for i in range(0, self.k)]
dataSet = [
    # 1
    [0.697, 0.460],
    # 2
    [0.774, 0.376],
    # 3
    [0.634, 0.264],
    # 4
    [0.608, 0.318],
    # 5
    [0.556, 0.215],
    # 6
    [0.403, 0.237],
    # 7
    [0.481, 0.149],
    # 8
    [0.437, 0.211],
    # 9
    [0.666, 0.091],
    # 10
    [0.243, 0.267],
    # 11
    [0.245, 0.057],
    # 12
    [0.343, 0.099],
    # 13
    [0.639, 0.161],
    # 14
    [0.657, 0.198],
    # 15
    [0.360, 0.370],
    # 16
    [0.593, 0.042],
    # 17
    [0.719,0.103],
        # 18
    [0.359,0.188],
        # 19
    [0.339,0.241],
        # 20
    [0.282,0.257],
        # 21
    [0.748, 0.232],
        # 22
    [0.714, 0.346],
        # 23
    [0.483, 0.312],
        # 24
    [0.478, 0.437],
        # 25
    [0.525, 0.369],
        # 26
    [0.751, 0.489],
        # 27
    [0.532, 0.472],
        # 28
    [0.473, 0.376],
        # 29
    [0.725, 0.445],
        # 30
    [0.446, 0.459],
]


model = K_means(dataSet)
model.Train_k_means()
error = model.MSE()
print(np.array(model.Get_k_class()))
x = []
y = []
color = []
plt.figure()
for i in range(0,len(model.Get_k_class())):
    for j in model.Get_k_class()[i]:
        x.append(j[0])
        y.append(j[1])
        color.append(i)
for i in range(0,len(model.Get_k_initialize())):
    x.append(model.Get_k_initialize()[i][0])
    y.append(model.Get_k_initialize()[i][1])
    color.append(i)

print(len(x))
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF','r'])
plt.scatter(x[0:30], y[0:30], c=color[0:30], edgecolors='k', s=50,cmap=cm_light)
plt.scatter(x[30:], y[30:], c=color[30:], edgecolors='k', s=50,marker='v',cmap=cm_light)
plt.show()