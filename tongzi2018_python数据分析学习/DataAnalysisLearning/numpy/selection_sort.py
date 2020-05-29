#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import numpy as np

def selection_sort(arr):
	for i in range(len(arr)):
		j = i + np.argmin(arr[i:])
		(arr[i],arr[j]) = (arr[j],arr[i])
	return arr

if __name__ == "__main__"	:
	x = [1,3,24,5,6]
	x = selection_sort(x)
	print(x)
