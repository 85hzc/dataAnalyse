《豆瓣电影分析_Python数据分析课设》
Douban-Movie-Crawler-Analysis
由豆瓣网站排行榜合法爬取2018年的电影排名信息，并对数据进行分析的项目

一、项目结构
1. Datas文件夹
RankByRegion：用csv文件来存放2018年各地区上映电影根据评分的排行。
RankByType：用csv文件来存放2018年各地区上映电影根据评分的排行。
SourceData：源数据文件，爬取的数据文件放在这里。数据分析时的源数据也是从这里读取。
2. src文件夹
spider_film：最初爬取所有电影的核心文件，爬取豆瓣上2018年所有的电影信息。并保存爬取到的信息到一个csv文件中。
spider_comment:爬取每种类型评分最高的电影的所有评论中的男女数量，以便后续作出每种类型男女比例的分析。
DataAna:数据预处理和数据分析的文件。导出2018年上映的电影中各类型和各地区的电影排名情况的csv文件、绘制图来显示数据的分析结果。

《Python数据分析学习》
项目介绍 最近在看数据分析，将自己的学习笔记记录下来，同时也分享给大家，恳请大家能够纠正错误和改进~~

注意：我这个系列写得比较乱，最近也没空更新，所以也烦请有兴趣的朋友提出建议，我把该系列整理好。 计划：后面我会将数据分析以jupyter notebook的形式发布，这样有助理解，所以后续不会持续更新单纯的代码了。

《dataplay2》
refer blog:https://my.oschina.net/taogang/blog/630632

Please refer to my blog(Chinese) for a simple introduction http://my.oschina.net/taogang/blog/630632

Add docker build @ https://github.com/gangtao/dataplay2/tree/master/docker in case you have trouble to run it.

cd dataplay2/docker
docker build -t dataplay:latest .
docker run -p 5000:5000 dataplay
or you can direcly run

docker run -p 5000:5000 naughtytao/dataplay
