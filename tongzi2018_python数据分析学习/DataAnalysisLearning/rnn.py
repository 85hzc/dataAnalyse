#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

#%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=.2)
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)


#逻辑斯特回归 Logistics Regression
clf = sklearn.linear_model.LogisticRegression()
clf.fit(X, y)

def plot_decision_boundry(pred_func):
     x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
     y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
     h = 0.01
     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
     Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
     Z = Z.reshape(xx.shape)
     plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
     plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)


plot_decision_boundry(lambda x: clf.predict(x))	 
plt.title('Logistic Regression')

plt.show()
