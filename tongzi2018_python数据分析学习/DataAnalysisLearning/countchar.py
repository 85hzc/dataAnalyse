
#-*- coding:utf-8 -*-

def countchar(string):
	string = string.lower()
	alphabet = "abcdefghijklmnopqrstuvwxyz"
	ans = {x:0 for x in alphabet}
	for s in string:
		if(s.isalpha()):
			ans[s] += 1
		
	return list(ans.values())

if __name__ == "__main__":
	s = input()
	print(countchar(s))
	