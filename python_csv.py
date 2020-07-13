# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:19:17 2020

@author: 28369
"""

#参考链接:
#https://www.cnblogs.com/qican/p/11122206.html    

#参考链接:
#https://www.cnblogs.com/cloud-ken/p/8432999.html

import csv
import codecs


# =============================================================================
# #data = [
# #    ("测试1",'软件测试工程师'),
# #    ("测试2",'软件测试工程师'),
# #    ("测试3",'软件测试工程师'),
# #    ("测试4",'软件测试工程师'),
# #    ("测试5",'软件测试工程师'),
# #]
# #f = codecs.open('test.csv','w','gbk')
# #writer = csv.writer(f)
# #for i in data:
# #    writer.writerow(i)
# #f.close()
# 
# f = csv.reader(open('test.csv', 'r'))
# for i in f:
# #    print(i)
#     for j in i:
#         print(j)
# 
# #['测试1', '软件测试工程师']
# #['测试2', '软件测试工程师']
# #['测试3', '软件测试工程师']
# #['测试4', '软件测试工程师']
# #['测试5', '软件测试工程师']
#     
#    
# #测试1
# #软件测试工程师
# #测试2
# #软件测试工程师
# #测试3
# #软件测试工程师
# #测试4
# #软件测试工程师
# #测试5
# #软件测试工程师    
# =============================================================================


import csv
import os
import numpy as np
import random
import requests
# name of data file
# 数据集名称
file = 'birth_weight.dat'
#birth_weight_file = 'birth_weight.csv'

## download data and create data file if file does not exist in current directory
## 如果当前文件夹下没有birth_weight.csv数据集则下载dat文件并生成csv文件
#if not os.path.exists(birth_weight_file):
#    birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
#    birth_file = requests.get(birthdata_url)
#    birth_data = birth_file.text.split('\r\n')
#    # split分割函数,以一行作为分割函数，windows中换行符号为'\r\n',每一行后面都有一个'\r\n'符号。
#    birth_header = birth_data[0].split('\t')
#    # 每一列的标题，标在第一行，即是birth_data的第一个数据。并使用制表符作为划分。
#    birth_data = [[float(x) for x in y.split('\t') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
#    print(np.array(birth_data).shape)
#    # (189, 9)
#    # 此为list数据形式不是numpy数组不能使用np,shape函数,但是我们可以使用np.array函数将list对象转化为numpy数组后使用shape属性进行查看。
#    with open(birth_weight_file, "w", newline='') as f:
#    # with open(birth_weight_file, "w") as f:
#        writer = csv.writer(f)
#        writer.writerows([birth_header])
#        writer.writerows(birth_data)
#        f.close()


#'单个dat文件转换为csv文件'
#def _dat_to_csv_(dat_path, dat_file):
#    file_r = open(dat_file, 'r')
#    csv_file = os.path.join(dat_path, dat_file.split('.')[0] + '.csv')
#    file_w = open(csv_file, 'w')
#    for line in file_r.readlines():
#         str_data = ",".join(line.split('\t'))
##        str_data = ",".join(line.split(','))
#         file_w.write(str_data)
#
#    file_r.close()
#    file_w.close()
#
#file = 'birth_weight.dat'
#_dat_to_csv_('./', file)

#exit()

#    
#import pandas as pd
#birth_weight_file = 'birth_weight.csv'
##f = csv.reader(open('001_cyt01.csv', 'r'))
#
#csv_data = pd.read_csv(birth_weight_file)
#print(csv_data.shape)  # (189, 9)
#N = 5
#csv_batch_data = csv_data.tail(N)  # 取后5条数据
#print(csv_batch_data.shape)  # (5, 9)
#print(csv_batch_data)
##     LOW  AGE  LWT  RACE  SMOKE  PTL  HT  UI   BWT
##184    0   31  120     0      0    0   0   0  4167
##185    0   35  170     0      0    1   0   0  4174
##186    0   19  120     0      1    0   1   0  4238
##187    0   24  216     0      0    0   0   0  4593
##188    0   45  123     0      0    1   0   0  4990
#
##train_batch_data = csv_batch_data[list(range(3, 6))]  # 取这20条数据的3到5列值(索引从0开始)
##print(train_batch_data)


    
#"""
#使用tensorflow读取csv文件
#"""
#import tensorflow as tf
#
#filename = 'birth_weight.csv'
#file_queue = tf.train.string_input_producer([filename])  # 设置文件名队列，这样做能够批量读取文件夹中的文件
#reader = tf.TextLineReader(skip_header_lines=1)  # 使用tensorflow文本行阅读器，并且设置忽略第一行
#key, value = reader.read(file_queue)
#defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]  # 设置列属性的数据格式
#LOW, AGE, LWT, RACE, SMOKE, PTL, HT, UI, BWT = tf.decode_csv(value, defaults)
## 将读取的数据编码为我们设置的默认格式
#vertor_example = tf.stack([AGE, LWT, RACE, SMOKE, PTL, HT, UI])  # 读取得到的中间7列属性为训练特征
#vertor_label = tf.stack([BWT])  # 读取得到的BWT值表示训练标签
#
## 用于给取出的数据添加上batch_size维度，以批处理的方式读出数据。可以设置批处理数据大小，是否重复读取数据，容量大小，队列末尾大小，读取线程等属性。
#example_batch, label_batch = tf.train.shuffle_batch([vertor_example, vertor_label], batch_size=10, capacity=100, min_after_dequeue=10)
#
## 初始化Session
#with tf.Session() as sess:
#    coord = tf.train.Coordinator()  # 线程管理器
#    threads = tf.train.start_queue_runners(coord=coord)
#    print(sess.run(tf.shape(example_batch)))  # [10  7]
#    print(sess.run(tf.shape(label_batch)))  # [10  1]
#    print(sess.run(example_batch)[3])  # [ 19.  91.   0.   1.   1.   0.   1.]
#    coord.request_stop()
#    coord.join(threads)
#
#'''
#对于使用所有Tensorflow的I/O操作来说开启和关闭线程管理器都是必要的操作
#with tf.Session() as sess:
#    coord = tf.train.Coordinator()  # 线程管理器
#    threads = tf.train.start_queue_runners(coord=coord)
#    #  Your code here~
#    coord.request_stop()
#    coord.join(threads)
#'''


"字典形式的读写"
#with open('name.csv', 'w') as csvfile:
#    datas = [[0, 5],
#             [1, 4],
#             [2, 13],
#             [3, 4],
#             [4, 5],
#             [5, 5],
#             [6, 15]]
#    writer = csv.DictWriter(csvfile, fieldnames=['id', 'class'])
#    # 写入列标题
#    writer.writeheader()
#    for data in datas:
#        writer.writerow({'id':data[0], 'class':data[1]})
        

#with open('name.csv') as csvfile:
#    # DictReader会将第一行内容作为key值，
#    # 第二行开始作为数据内容
#    reader = csv.DictReader(csvfile)
##    print(reader)
#    for row in reader:
#        print(row)
#        print(row['id'], row['class'])

#OrderedDict([('id', '0'), ('class', '5')])
#0 5
#OrderedDict([('id', '1'), ('class', '4')])
#1 4
#OrderedDict([('id', '2'), ('class', '13')])
#2 13
#OrderedDict([('id', '3'), ('class', '4')])
#3 4
#OrderedDict([('id', '4'), ('class', '5')])
#4 5
#OrderedDict([('id', '5'), ('class', '5')])
#5 5
#OrderedDict([('id', '6'), ('class', '15')])
#6 15


"读取csv文件"
with open('name.csv', 'r') as f:
    reader = csv.reader(f)
#    print(type(reader))
    
#    result = list(reader)
#    print(result[1]) # 获取某一行
    
#    for i in reader:
#        print(i[0]) # 获取某一列
    
    
#    # 以列表的形式输出每一行
#    for row in reader:
#        print(row) 

#['id', 'class']
#[]
#['0', '5']
#[]
#['1', '4']
#[]
#['2', '13']
#[]
#['3', '4']
#[]
#['4', '5']
#[]
#['5', '5']
#[]
#['6', '15']
#[]



















    
    