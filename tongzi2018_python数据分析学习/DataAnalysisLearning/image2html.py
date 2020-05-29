#/usr/bin/env python3
#-*- coding: utf-8 -*-
#

#installation: pip install git+https://github.com/lzjun567/img2html
import img2html
converter = img2html.converter.Img2HTMLConverter(char='傻')
path = './cai.jpg'
html = converter.convert(path)

with open('cai.html', mode='w', encoding='utf-8') as f:
	f.write(html)