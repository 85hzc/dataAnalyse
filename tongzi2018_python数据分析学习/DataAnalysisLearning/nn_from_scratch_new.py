#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#Name：     module
#Purpose：  Neural Network
#
#Author：   tongzi
#
#Created：  2018/1/28
#Copyright: (c)tongzi 2018
#License:    tongzi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import sklearn.linear_model
import sklearn
from sklearn import datasets
#import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.2)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#clf = sklearn.linear_model.LogisticRegressionCV()
#clf.fit(X, y)
#plt.show()

def plot_decision_boundary(pre_func):
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pre_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)
	
'''    
plot_decision_boundary(lambda x: clf.predict(x))
plt.title('Logistic Regression')
plt.show()
'''

num_examples = len(X) #训练集的个数，当前是200个
nn_input_dim = 2      #输入层的维度
nn_output_dim = 2     #输出层的维度


#梯度下降的参数
epsilon = 0.01        #学习率
reg_lambda = 0.01     #正则化的强度

def calculate_loss(model):
	#print('inside model...')
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	#print('Test...')
	#前向传播计算估计值
	z1 = X.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	
	#计算损失函数
	correct_logprobs = -np.log(probs[range(num_examples), y])
	data_loss = np.sum(correct_logprobs)
	
	#正则化（可选）
	data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	return 1. / num_examples * data_loss
	

def nn_predict(model, x):
	#print('inside nn_predict...')
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	
	#前向传播计算估计值
	z1 = x.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	return np.argmax(probs, axis=1)
	

def build_model(nn_hdim, num_passes=20000, print_loss=False):
	#print('inside build_model')
	#参数初始化
	np.random.seed(0)
	W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
	b1 = np.zeros((1, nn_hdim))
	W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
	b2 = np.zeros((1, nn_output_dim))
	
	#最终要返回的值
	model = {}
	for i in range(0, num_passes):
		#print('inside for ')
		#前向传播
		z1 = X.dot(W1) + b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2) + b2
		exp_scores = np.exp(z2)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		
		#反向传播
		delta3 = probs
		delta3[range(num_examples), y] -= 1
		dW2 = (a1.T).dot(delta3)
		db2 = np.sum(delta3, axis=0, keepdims=True)
		delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
		dW1 = np.dot(X.T, delta2)
		db1 = np.sum(delta2, axis=0)
		
		#正则化W1，W2，b1，b2不需正则化
		dW2 += reg_lambda * W2
		dW1 += reg_lambda * W1
		
		#梯度下降参数更新
		W1 += -epsilon * dW1
		b1 += -epsilon  * db1
		W2 += -epsilon * dW2
		b2 += -epsilon * db2
		
		model = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
		if print_loss and i % 1000 == 0:
			print('Loss after iteration %i: %f'%(i, calculate_loss(model)))
			
	return model		
		
'''
model = build_model(3, print_loss=True)
fig = plt.subplots();
plot_decision_boundary(lambda x: nn_predict(model, x))
plt.title('Decision boundary for hidden layer size 3')
plt.show()
'''

plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1,2,3,4,5,20,50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
	plt.subplot(5, 2, i+1)
	plt.title('Hidden Layer size %d' % nn_hdim)
	model = build_model(nn_hdim)
	plot_decision_boundary(lambda x: nn_predict(model, x))
plt.show()

	
	











