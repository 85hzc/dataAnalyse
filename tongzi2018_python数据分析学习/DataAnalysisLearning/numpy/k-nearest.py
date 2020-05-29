#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn

seaborn.set()

rand = np.random.RandomState(42)
data = rand.rand(10, 2)

plt.scatter(data[:,0],data[:,1], s=100)

#求两点间距离的平方
#np.sum(axis=-1)：axis=-1表示对矩阵的最大的轴进行求和，举例：
#bb = np.arange(24).reshape((4,6)),输出：
'''
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23]])
'''
#np.sum(bb,axis=-1)的输出：
'''
array([ 15,  51,  87, 123])
'''
#
#
dist_square = np.sum((data[:,np.newaxis,:] - data[np.newaxis,:,:])**2, axis=-1)

#对每行进行排序。np.argmin()返回每个元素在排序后的
#矩阵（或向量）中的索引
nearest = np.argmin(dist_square, axis=1)

k = 2
#返回前k个最小值得索引
k_nearest = np.argpartition(dist_square, k+1, axis=1)

for i in range(data.shape[0]):
	for j in k_nearest[i,:k+1]:
		plt.plot(*zip(data[i],data[j]), color='black')

plt.show()









