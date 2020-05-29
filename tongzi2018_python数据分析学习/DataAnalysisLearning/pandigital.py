
def pandigital(nums):
	ans = []
	for n in nums:
		lenght = len(str(n))
		flag = False 
		for x in range(1,lenght+1):
			if(str(x) not in str(n)):
				flag = True
				break
			
		if(flag == False):
			ans.append(n)
	return ans
	
if __name__ == "__main__":
	lst = pandigital(eval(input()))
	if(0 == len(lst)):
		print('not found')
	else:
		for x in lst:
			print(x)
	
	
	
	
	
			
	
	
	
	