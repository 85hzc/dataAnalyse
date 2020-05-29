
with open('test.txt', 'r+') as f:
	f.seek(15)
	print(f.readline())
with open('test.txt') as f:
	data = []
	for d in f:
		data.append(d)
print(data)