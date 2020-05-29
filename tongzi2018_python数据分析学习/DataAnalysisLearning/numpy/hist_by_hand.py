#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42) #设置随机种子
x = np.random.randn(100) #产生100符合均值为0，方差为1的正太分布数据

#手动计算直方图
bins = np.linspace(-5, 5, 20) #直方图的区间
counts = np.zeros_like(bins) #产生一个形状（维度）和bins相同，值全为0的矩阵

#为x中的每个元素在bins中找出其所在的索引
i = np.searchsorted(bins, x)

#为每个区间加1
np.add.at(counts, i, 1)

plt.plot(bins, counts, linestyle='steps')
plt.show()