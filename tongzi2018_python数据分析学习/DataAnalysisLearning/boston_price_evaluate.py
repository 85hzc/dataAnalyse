#!/usr/bin/evn python3
#-*- coding: utf-8 -*-


import numpy as np  #导入numpy库
import pandas as pd #导入pandas库
import matplotlib.pyplot as plt #导入matplotlib绘图库
import seaborn as sns  #导入seaborn绘图库

sns.set() #设置seaborn绘图风格

from sklearn import datasets  #引入datasets命名空间
import statsmodels.api as sm #导入线性回归模型

#获取Boston房价数据
boston = datasets.load_boston()

#建立DataFrame
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=['MEDV'])

def ols_model():

	#statsmodels中的线性回归模型没有截距，因此给训练数据加上一列全为1的特征
	X_add_constant = sm.add_constant(X)

	#选择模型
	ols = sm.OLS(y, X_add_constant) #选择最小二乘模型

	#使用模型进行拟合
	model = ols.fit()

	#查看拟合之后的综合信息,去除显著性特征P大于0.05的列
	print(model.summary())

	#去除显著性特征P大于0.05的列
	#注意：axis必须指定为1，否则删除的就是行数据
	#inplace指示对X的修改会反应到源数据，也就是就地修改
	X.drop('AGE', axis=1, inplace=True)
	X.drop('INDUS', axis=1, inplace=True)


	#statsmodels中的线性回归模型没有截距，因此给训练数据加上一列全为1的特征
	X_add_constant = sm.add_constant(X)
	#去除不合的列之后，重新训练
	#选择模型
	ols = sm.OLS(y, X_add_constant) #选择最小二乘模型

	#使用模型进行拟合
	model = ols.fit()

	#查看拟合之后的综合信息,去除显著性特征P大于0.05的列
	print(model.summary())

	#测试集预测
	X_test = np.array([[1, 2, 18.0, 3, 5, 6.6, 8, 1.0, 290.0,
	15.2, 396.2, 12]]) 
	print(model.predict(X_test))


	
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
	
def linear_model():
	model = LinearRegression(fit_intercept=False)#fit_intercept=False表示不在模型中使用截距
	model.fit(X, y)
	ans_train = model.predict(X)
	y['predicted'] = ans_train
	y[['MEDV', 'predicted']].plot(alpha=0.5) #对比真实的和预测的结果
	
	#评估各个特征的影响
	
	params = pd.DataFrame( model.coef_, columns = X.columns)
	print("params:\n",params)
	#不确定性评估
	np.random.seed(1)
	err = np.std([model.fit(*resample(X, y)).coef_ for i in range(1000)], 0)
	print("err:\n",err)
	#print(pd.DataFrame({'effect': params,'error':err}))
	plt.show()
	
if __name__ == "__main__":
	#ols_model()
	linear_model()

