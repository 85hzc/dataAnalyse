#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import pandas as pd
url = 'https://developers.douban.com/wiki/?title=book_v2'
def make_df(cols, ind):
	data = {c: [str(c) + str(i) for i in ind] for c in cols}
	return pd.DataFrame(data,ind)
	
if __name__ == "__main__":
	print(make_df(['apple','peach','pear'],range(3)))