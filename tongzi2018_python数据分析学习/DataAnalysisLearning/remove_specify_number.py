#!/usr/bin/env python3
#-*- coding:utf-8 -*-

def find_numbers(n_min, n_max, n):
	j = 0
	for i in range(n_min, n_max+1):
		if(not((str(n) in str(i)) or (i % n == 0))):
			j = j + 1
			if((j) % 10 == 0):
				print(i,end=' ')
			else:
				print(i,end=',')
			
			
		if(j % 10 == 0):
			print()
		

if __name__ == "__main__":
	n_min = 1
	n_max = 100
	n = input()
	n = int(n)
	find_numbers(n_min, n_max, n)