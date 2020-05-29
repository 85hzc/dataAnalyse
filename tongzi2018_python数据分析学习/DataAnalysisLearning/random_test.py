#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import numpy as np

n = 500
ans = []

j = 0
with open('random.txt','w') as f:
	
	for i in range(n):
		
		j = j + 1
		num = np.random.randint(1,101)
		ans.append(num)
		if(j % 10 == 0):
			f.write(str(num) + "\n")
		else:
			f.write(str(num) + ",")
	

	
#计算众数
res = {}

for num in ans:
	if(num not in res):
		res[num] = 1
	else:
		res[num] += 1
#print(max(res.values()))
res_sorted = sorted(res.items(), key=lambda x:x[1], reverse=True)
#print(res_sorted)
print(res_sorted[0][0])


	