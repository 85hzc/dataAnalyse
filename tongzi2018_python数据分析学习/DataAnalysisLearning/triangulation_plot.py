#!/usr/env/bin python3
#-*-coding:utf-8-*-

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def f(x, y):
	return np.sin(np.sqrt(x**2 + y**2))

theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))#降维：转为一维
y = np.ravel(r * np.cos(theta))
z = f(x, y)

ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)

#用三角形填充曲面
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')

plt.show()
