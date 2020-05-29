#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
#@description:3d plot
#@author:tongzi
#@date:2018/11/21 11:28
#@version:1.0
#@license:No
#
'''

import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0,8.0)

sns.set()

data = pd.read_csv('data_points.csv') #读取csv文件
data.columns = ['x', 'y', 'val'] #重置列名为x，y，val

#画3D图形
fig1 = plt.subplots()
ax1 = plt.axes(projection='3d')#指定参数projection='3d'，画三维图
ax1.plot3D(data['x'], data['y'], data['val'])

ax1.set_title('Magnetic Distribution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z/Frequency Value')

#画X-Z平面的投影
fig2 = plt.subplots()
ax2 = plt.axes()
ax2.plot(data['x'], data['val'])

ax2.set_title('X-Z Projection')
ax2.set_xlabel('x')
ax2.set_ylabel('y/Frequency Value')


#画Y-Z平面的投影
fig3 = plt.subplots()
ax3 = plt.axes()
ax3.plot(data['y'], data['val'])

ax3.set_title('Y-Z Projection')
ax3.set_xlabel('x')
ax3.set_ylabel('y/Frequency Value')

#画X-Y平面，采集区域
fig4 = plt.subplots()
ax4 = plt.axes()
ax4.plot(data['x'], data['y'])

ax4.set_title('X-Y Collection Area')
ax4.set_xlabel('x')
ax4.set_ylabel('y')


#绘制曲面
x = data['x'].drop_duplicates() #去掉重复值
y = data['y'].drop_duplicates()

zz = data['val'].reshape(len(x),len(y))
xx, yy = np.meshgrid(x, y)

fig5 = plt.subplots();
ax5 = plt.axes(projection='3d')
ax5.plot_surface(xx, yy, zz,cmap=plt.cm.hot)





plt.draw() #强制绘制
plt.show() #显示图形

