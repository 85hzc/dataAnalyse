#!/usr/bin/env python3
#-*- coding:utf-8 -*-

'''
BeatifukSoup的使用
抓取豆瓣书评中《小王子》的评论
'''

import requests
from bs4 import BeautifulSoup

url = "https://book.douban.com/subject/1084336/comments/"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')
pattern = soup.find_all("span",'short')
#print(pattern)
for item in pattern:
	print(item.string)