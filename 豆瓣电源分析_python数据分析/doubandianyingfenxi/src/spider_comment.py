# This py file was written by another group member

import requests
import time
import re
from fake_useragent import UserAgent

ua = UserAgent()


# 获取评论用户索引页
def get_name_list(html):
    try:
        headers = {'User-Agent': ua.random}
        response = requests.get(url=html, headers=headers)
        time.sleep(5)
        response.encoding = 'utf-8'
        print('//////////////')
        print(response)
        if response.status_code == 200:
            return response.text
        return None
    except requests.exceptions:
        print('获取用户索引页错误')
        time.sleep(3)
        return get_name_list(html)


# 解析评论用户索引页
def parse_user_name_list(html):
    #str(html).strip()
    #str(html).replace('\n','')
    #str(html).replace('\r','')
    #print(html)
    html = get_name_list(html)#response comment page
    #print(html)//debug response
	
    #获取影片名
    # 获取电影名称
    name_pattern = re.compile('<span property="v:itemreviewed">(.*?)</span>')
    filminfo = re.findall(name_pattern, str(html))
    filmname = set(filminfo)
    print('film[%s]'%(filmname))
    # 获取评论用户名
    name_pattern = re.compile('https://www.douban.com/people/(.*?)/')
    name = re.findall(name_pattern, str(html))
    # 去重
    print('parse_user_name_list [%s]'%name)
    user_name_list.append(list(set(name)))


# 获取评论用户主页
def get_name_page(html):
    try:
        headers = {'User-Agent': ua.random}
        response = requests.get(url=html, headers=headers)
        response.encoding = 'utf-8'
        if response.status_code == 200:
            return response.text
        return None
    except requests.exceptions:
        print('获取用户主页错误')
        time.sleep(3)
        return get_name_page(html)


# 解析评论用户主引页
def parse_name_page(html):
    html = get_name_page(html)
    # 获取评论用户名
    sex_pattern = re.compile('<span class="gender (.*?)" />')
    sex = re.findall(sex_pattern, str(html))
    sex_list.append(sex)


user_name_list = []
sex_list = []
subject_list = []

file=open('list.txt','r')
lines = file.readlines()      #读取全部内容 ，并以列表方式返回  
for line in lines:
    print('append '+line)
    subject_list.append(line)
file.close()

for prefix in subject_list:
    pre = str(prefix).strip()
    #prefix.strip()
    print('parse user name >>>>>>>>>>>'+pre)
    user_name_list.clear()
    for i in range(0, 200, 20):
        parse_user_name_list(pre+"comments?start=" + str(i) + "&limit=20&sort=new_score&status=P&percent_type=h")
        print(i)

    j=0
    sex_list.clear()
    for temp in user_name_list:
        print('peple '+'%s'%str(temp))
        #for temp2 in temp:
        print('spider_comment parse_name_page peple [%d]'%j+'%s'%str(temp))
        parse_name_page("https://m.douban.com/people/" + str(temp) + "/")
        j = j + 1
        time.sleep(1)

    male = 0
    female = 0
    for i in sex_list:
        for j in i:
            for k in j:
                if (k == 'm'):
                    male += 1
                elif (k == 'f'):
                    female += 1
    print(filmname)
    print("男生人数 : ", male)
    print("女生人数 : ", female)

