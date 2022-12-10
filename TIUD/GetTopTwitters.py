#!/usr/bin/python2
# -*-coding:utf-8-*-
import re
import urllib2

url = 'http://twitaholic.com/'
id = '@(.*)<br />'  
id_re = re.compile(id) 

res = urllib2.urlopen(url)
data = res.read()
id_info = id_re.findall(data)

f = open('result.txt', 'w+')

for item in id_info:
    print item
    f.write(item + '\n')