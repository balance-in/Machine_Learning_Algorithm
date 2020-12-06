from libsvm.svmutil import *
from libsvm.svm import *
from matplotlib.pylab import mpl
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

data = [[1,1,1],[1,0,1],[2.5,3.3,1],
        [3,3,-1],[3,4,-1],[3.5,3,-1]]
x, y = np.split(data, (2,), axis=1)
clf = svm.SVC(C=10, kernel='linear', decision_function_shape='ovo')

clf.fit(x, y.ravel())

x1_min, x1_max = 0, 5
x2_min, x2_max = 0, 5
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
grid_test = np.stack((x1.flat, x2.flat), axis=1)
grid_hat = clf.predict(grid_test)
grid_hat = grid_hat.reshape(x1.shape)
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
fig=plt.figure(figsize=(8,8))
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)

plt.scatter(x[:, 0][0:3], x[:, 1][0:3], c=['red'],edgecolors='k', s=50, cmap=cm_dark)# 样本
plt.scatter(x[:, 0][3:6], x[:, 1][3:6], c=['green'],edgecolors='k', marker='v',s=50, cmap=cm_dark)
#plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本


plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

plt.savefig("SVM_C=10",dpi=300)
plt.show()
