# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:08:37 2020

@author: 28369
"""



import numpy as np


"Pythonic code"
# =============================================================================
# # 两个变量交换
# a = 2
# b = 3
# a, b = b, a
# print(a, b)
# =============================================================================

# =============================================================================
# #翻转
# a = [1, 2, 3, 4]
# c = "abcdef"
# 
# # 不推荐
# print(a[::-1])
# print(c[::-1])
# 
# # 推荐使用Python库里reversed()函数
# print(list(reversed(a)))
# print(list(reversed(c))
# =============================================================================

# =============================================================================
# #字符串格式化
# print('{greet} from {language}.'.format(greet = 'Hello World', 
#                                         language = 'Python'))
# 
# =============================================================================

# =============================================================================
# # 利用assert语句来发现问题
# x = 1
# y = 2
# assert x == y, "not equals"
# =============================================================================


# =============================================================================
# # 重复利用Lazy evaluation 的特性
# from itertools import islice
# def fib():
#     a, b = 0, 1
#     while True:
#         yield a
#         a, b = b, a+b
#         
# print(list(islice(fib(), 5)))
# =============================================================================

# yield 用法

#def foo():
#    print("starting...")
#    while True:
#        res = yield 4 
#        print("res:",res)
#g = foo()
#print(next(g))
#print("*"*20)
#print(next(g))

#starting...
#4
#********************
#res: None
#4



#def foo():
#    print("starting...")
#    while True:
#        res = yield 4
#        print("res:",res)
#        
#g = foo()
#print(next(g))
#print("*"*20)
#print(g.send(7))

#starting...
#4
#********************
#res: 7
# 4

## 适用于遍历大的范围, 节省内存空间
#def foo(start, end):
#    print("starting...")
#    while start < end:
#        start = start+1
#        yield start
#
#for n in foo(0, 4):
#    print(n)
#
##starting...
##1
##2
##3
##4




# =============================================================================
# # 不推荐使用type来进行类型检查
# import types
# class UserInt(int):
#     def __init__(self, val=0):
#         self._val = int(val)
#     def __add__(self, val):
#         if isinstance(val, UserInt):
#             return UserInt(self._val + val._val)
#     def _iadd__(self, val):
#         raise NotImplementedError("not support operation")
#     def __str__(self):
#         return str(self._val)
#     def __repr__(self):
#         return 'Integer(%s)' %self._val
#     
# n = UserInt()
# print(n)
# m = UserInt(2)
# print(m)
# print(n+m)
# 
# print(isinstance(2, float))
# print(isinstance("a", (str)))
# =============================================================================


# =============================================================================
# # 浮点数的比较同样最好能够指明精度
# i = 1
# while i != 1.5:
#     i = i + 0.1
#     if i > 2:
#         break
#     print(i)
#     
# i = 1.0
# while abs(i - 1.5) > 0.01:
#     i += 0.1
#     if i >2:
#         break
#     print(i)
# =============================================================================


# 使用enumerate() 获取序列迭代的索引和值
#li = ['a', 'b', 'c', 'd', 'e']
#for i, e in enumerate(li):
#    print("index:", i, "element:", e)
#
#e = enumerate(li)

#print(e)

#index: 0 element: a
#index: 1 element: b
#index: 2 element: c
#index: 3 element: d
#index: 4 element: e




##enumerate 实际相当于如下代码:
#def enumerate(sequence, start=0):
#    n = start
#    for elem in sequence:
#        yield n, elem
#        n += 1
#
##利用这个特性可以自己设计一个enumerate()函数, 
##比如：以反序的方式获取序列的索引和值
#def myenumerate(sequence):
#    n = -1
#    for elem in reversed(sequence):
#        yield len(sequence)+n, elem
#        n -= 1
#for i, e in myenumerate(li):
#    print("index:", i, "element:", e)


# =============================================================================
# #分清==与is的使用场景
# #is 表示的是对象标示符，比较的是两个对象在内存中是否拥有同一块内存空间,
# #==表示的是相等，用来检验两个对象的值是否相等
# # 当两个基本类型数据(元组)内容相同时, id会相同, 但并不代表a会随b的改变而改变
# import copy
# str1 = "we are family"
# str2 = copy.copy(str1)
# 
# print(str1 is str2)
# =============================================================================

# 考虑兼容性, 尽可能使用Unicode
# Unicode(Universal Multiple-Octect Coded Character Set) 万国码
# 其实现方式称为Unicode转换格式UTF(Unicode Transformation Format)
# UTF-8的特点是对不同范围的字符使用不同长度的编码
#strUnicode = u"unicode 字符串"     # 前面加u表示Unicode
#print(strUnicode)






"dict()"

#key = dict(a='a', b='b', t='t')     # 传入关键字
#map_func = dict(zip(['one', 'two', 'three'], [1, 2, 3]))   # 映射函数方式来构造字典
#iterable = dict([('one', 1), ('two', 2), ('three', 3)])    # 可迭代对象方式来构造字典
#
#print(key)
#print(map_func)
#print(iterable)
#
##{'a': 'a', 'b': 'b', 't': 't'}
##{'one': 1, 'two': 2, 'three': 3}
##{'one': 1, 'two': 2, 'three': 3}




    
"""
装饰器
"""
#
#class Student(object):
#    def __init__(self, name, score):
#        self.name = name
#        self.score = score
#        
##当我们想要修改一个 Student 的 scroe 属性时, 可以这么写: 
#s = Student('Bob', 59)
#s.score = 60
##但是也可以这么写: 
#s.score = 1000
##显然, 直接给属性赋值无法检查分数的有效性。
##如果利用两个方法: 
#class Student(object):
#    def __init__(self, name, score):
#        self.name = name
#        self.__score = score
#    def get_score(self):
#        return self.__score
#    def set_score(self, score):
#        if score < 0 or score > 100:
#            raise ValueError('invalid score')
#        self.__score = score
#        
##这样一来, s.set_score(1000) 就会报错。
##这种使用 get/set 方法来封装对一个属性的访问在许多面向对象编程的语言中都很常见。
##但是写 s.get_score() 和 s.set_score() 没有直接写 s.score 来得直接。
##有没有两全其美的方法？----有。
##因为Python支持高阶函数, 可以用装饰器函数把 get/set 方法“装饰”成属性调用: 
#
#class Student(object):
#    def __init__(self, name, score):
#        self.name = name
#        self.__score = score
#    @property
#    def score(self):
#        return self.__score
#    @score.setter
#    def score(self, score):
#        if score < 0 or score > 100:
#            raise ValueError('invalid score')
#        self.__score = score
#        
##注意: 第一个score(self)是get方法, 
##用@property装饰, 第二个score(self, score)是set方法, 
##用@score.setter装饰, @score.setter是前一个@property装饰后的副产品。
##现在, 就可以像使用属性一样设置score了: 
#
#s = Student('Bob', 59)
#s.score = 60
#print(s.score)  # 60
#s.score = 1000
#print(s.score)
##Traceback (most recent call last):
##  ...
##ValueError: invalid score
#
##说明对 score 赋值实际调用的是 set方法。

#@g
#@f
#def foo():
#    :
#等价于 foo = g(f(foo))

# 修饰符举例
#from time import ctime, sleep
#
#def tsfunc(func):
#    def wrappedFunc():
#        print("[%s] %s() called" % (ctime(), func.__name__))
#        return func()
#    return wrappedFunc
#
#@tsfunc
#def foo():
#    pass
#
#foo()
#sleep(4)
#
#for i in range(2):
#    sleep(1)
#    foo()

#[Fri Nov 22 14:58:26 2019] foo() called
#[Fri Nov 22 14:58:31 2019] foo() called
#[Fri Nov 22 14:58:32 2019] foo() called


"删除列表指定元素"
#deleK = 3
#L = [1, 3, 4, 12, 3, 3, 34, 23, 12, 3]
#L = filter(lambda x: x != deleK, L)
#L = [i for i in L]
#print(L)  #[1, 4, 12, 34, 23, 12]

#L_filter = [x for x in L if x!=3]
#print(L_filter)  [1, 4, 12, 34, 23, 12]

"赋多个值"
#L = [0]*5
#print(L)  # [0, 0, 0, 0, 0]

#L = [[0]] * 5 
#print(L)  # [[0], [0], [0], [0], [0]]

"移除列表指定位置的元素"
#num_list = [12,4,56,8,0,34,6,44, 0, 5, 0, 2]
#num_list.pop(3)
#print(num_list)  # [12, 4, 56, 0, 34, 6, 44, 0, 5, 0, 2]

"在指定位置插入元素"
#list = [1, 2, 4, 5]
#list.insert(2, 3)
#print(list)  # [1, 2, 3, 4, 5]

"生成稀疏矩阵"
#from scipy import sparse

#row = [2,2,3,2]
#col = [3,4,2,5]
#data = [1,1,1,1]
#c = sparse.csr_matrix((data, (row, col)), shape=(5, 6))
#print(c)
#print(c.toarray())
##  (2, 3)        1
##  (2, 4)        1
##  (2, 5)        1
##  (3, 2)        1
##[[0 0 0 0 0 0]
## [0 0 0 0 0 0]
## [0 0 0 1 1 1]
## [0 0 1 0 0 0]
## [0 0 0 0 0 0]]

#indptr = np.array([0, 2, 3, 6])
#indices = np.array([0, 2, 2, 0, 1, 2])
#data = np.array([1, 2, 3, 4, 5, 6])
#mat = sparse.csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
#print(mat)
##[[1 0 2]
## [0 0 3]
## [4 5 6]]

#scipy.sparse.rand(m,n,density,format,dtype,random_state)
#m,n	整型；表示矩阵的行和列
#density	实数类型；表示矩阵的稀疏度
#format	str类型；表示矩阵的类型；如format=‘coo’
#dtype	dtype;表示返回矩阵值的类型
#ranom_state	{numpy.random.RandomState,int};可选的随机种子；如果空缺, 默认numpy.random

#n=4
#m=4
#density=0.5
#matrixformat='coo'
#B=sparse.rand(m,n,density=density,format=matrixformat,dtype=None)
#print(B.toarray())



"调用不同文件夹下的.py文件"
#假设A.py文件的文件路径为: E:/PythonProject/winycg
#那么B.py文件如下调用A.py文件

#python import模块时,  是在sys.path里按顺序查找的。
#sys.path是一个列表, 里面以字符串的形式存储了许多路径。
#使用A.py文件中的函数需要先将他的文件路径放到sys.path中

#import sys
#sys.path.append(r'E:/PythonProject/winycg')
#
#import A

# 示例
#import sys
# 方式一
#sys.path.append(r"C:/Users/SXY/.spyder-py3/")
#import MG.exercise_1 as me
# 方式二
#sys.path.append(r"C:/Users/SXY/.spyder-py3/MG")
#from exercise_1 import Matrix_Convert_CSR_Diagonal_Priority
#A = np.matrix([[1, 0, 2],
#               [0, 2, 0],
#               [0, 1, 1]])
#B = np.matrix([[2, 2, 3],
#               [4, 3, 2],
#               [-1, -1, 3]])

# 对应方式一
#AA, JA, IA = me.Matrix_Convert_CSR_Diagonal_Priority(A)
# 对应方式二
#AA, JA, IA = Matrix_Convert_CSR_Diagonal_Priority(A)

#print("AA:", AA)            
#print("JA:", JA)      
#print("IA:", IA)

#AA: [1, 2, 2, 1, 1]
#JA: [0, 2, 1, 2, 1]
#IA: [0, 2, 3, 5]

"if and"
#x and y 的值只能是x或y. x为真就是y, x为假就是x. 
#print(5 and 4)  # 4 
#x or y 的值只可能是x或y. x为真就是x, x为假就是y.
#print(5 or 4) # 5

#if 5>4 & 5<10:
#    print("true") # true
#if 5>4 and 5<10: 
#    print("true") # true

#if 5>4 and 5>10:
#    print("true")
#else:
#    print("false")

#false

#print('' or 5 or 0) # 5
#print(0 or 5 and 7 or 0) # 7

"sys.argv"

#"test.txt"文档包含的内容:
#this is the first line content typing in...
#I input the content from keyboard and prepare to write into file. 
#this is the third line
#this is the fourth line
#this is the fifth line  

#以下代码保存为"exe.py"文件
#from sys import argv
## 用户输入的参数, 这里输入 "exe.py test.txt"
#script,input_file = argv  
#def print_all(f):
#	print (f.read())
#	
#def rewind(f):
#	f.seek(2) # this 这个词从第2个词i开始输出
#	
#def print_a_line(line_count,f):
#	print (line_count,f.readline())
#	
#current_file = open(input_file)
# 
#print ("First let's print the whole file:\n")
# 
#print_all(current_file)
# 
#print ("Now let's rewind,kind of like a tape.")
# 
#rewind(current_file)
# 
#print ("Now let's print three lines.")
# 
#current_line = 1
#print_a_line(current_line,current_file)
# 
#current_line = current_line + 1
#print_a_line(current_line,current_file)
# 
#current_line = current_line + 1
#print_a_line(current_line,current_file)

#输出结果:
#First let's print the whole file:
#
#this is the first line content typing in...
#I input the content from keyboard and prepare to write into file.
#this is the third line
#this is the fourth line
#this is the fifth line
#Now let's rewind,kind of like a tape.
#Now let's print three lines.
#1 is is the first line content typing in...  
#
#2 I input the content from keyboard and prepare to write into file.
#
#3 this is the third line


"seek()"
#seek(offset, whence=0)
#offset: 开始的偏移量, 也就是代表需要移动偏移的字节数
#whence: 给offset参数一个定义, 表示要从哪个位置开始偏移；
#0代表从文件开头开始算起, 
#1代表从当前位置开始算起, 
#2代表从文件末尾算起。

#f = open("test.txt", "r+")  # "r+"表示随时都可进行读与写
#f.seek(0, 2)
#f.write("commit;")
#f.close()

"join"
#语法:'sep'.join(seq)
#sep：分隔符。可以为空
#seq：要连接的元素序列、字符串、元组、字典
#上面的语法即：以sep作为分隔符，将seq所有的元素合并成一个新的字符串
#
#返回值：返回一个以分隔符sep连接各个元素后生成的字符串

#seq_1 = ['My','name','is','family']
#print(" ".join(seq_1))
#print(":".join(seq_1))
##My name is family
##My:name:is:family
#
#seq_2 = "My name is family"
#print(":".join(seq_2))
##M:y: :n:a:m:e: :i:s: :f:a:m:i:l:y
#seq_3 = ('My','name','is','family')
#print(":".join(seq_3))
##My:name:is:family
#seq_4 = {'My':1,'name':2,'is':3,'family':4}
#print(":".join(seq_4))
##My:name:is:family

"__future__"

#PEPs (Python Enhancement Proposals) 官网:
#https://www.python.org/dev/peps/
#由于Python是由社区推动的开源并且免费的开发语言，不受商业公司控制，
#因此，Python的改进往往比较激进，不兼容的情况时有发生。
#Python为了确保你能顺利过渡到新版本，
#特别提供了__future__模块，让你在旧的版本中试验新版本的一些特性。
#对于所有的from __future__ import _，
#意味着在新旧版本的兼容性方面存在差异，处理方法是按照最新的特性来处理
#例如:
##确保为绝对导入
#from __future__ import absolute_import
#
##在Python2.x代码中直接使用Python3.x的精确除法
##例如 10/3 表示精确除法, 结果为 3.3333333... 
##10 // 3 表示截断除法, 结果为 3
#from __future__ import division
#
##即使在低版本的python2.X，当使用print函数时，须像python3.X那样加括号使用
#from __future__ import print_function


"isinstance(object, class-or-type-or-tuple)"
#调用python内置函数时，如果参数个数不对或参数类型不对，python解释器会自动检查出来；
#而调用自己写的函数时，python可以检查出参数个数不对，但不能检查出参数类型
#数据类型检查可以使用内置函数isinstance()实现
#内置函数isinstance有两个参数，
#第一个参数是待检测的对象，第二个参数是对象类型，
#可以是单个类型，也可以是元组，返回的是bool类型

#def my_abs(x):
#     if not isinstance(x, (int, float)):
#        raise  TypeError('bad operand type')
#     if x>=0:
#        return x
#     else:
#         return -x
# 
##my_abs('A') #TypeError: bad operand type
#print(my_abs(-3))
#print(isinstance('adf', (str, float, int)))
##3
##True

"python列表容易出错的地方"
s = [1, 4, 6, 12, 12, 8 , 9, 6, 9]
#for i in s:
#  if (i%3==0) and (i%2==0):
#    s.remove(i)
#print(s) # 其中有一个12没有被删除
#[1, 4, 12, 8, 9, 9]

#s = [i for i in s if not (i%3==0 and i%2 == 0)] # 可解决上述问题
#print(s) #[1, 4, 8, 9, 9]


"extend"
#s = [1, 2, 3, 4]
#q = [1, 2, 3, 4]
#s.append([0]*4)
#q.extend([0]*4)
#print(s)
#print(q)
##[1, 2, 3, 4, [0, 0, 0, 0]]
##[1, 2, 3, 4, 0, 0, 0, 0]
#t = [1, 2, 3, 4]
#t.extend([1,2]*-2)
#print(t)
#t.remove(2)
#print(t)

#s = ['1', '2', '3', '4']
#print(s)
#s = np.array(s).reshape(2, 2)
#print(s)
#s = s.tolist()
#print(s)

#s = ['1', '2', '3', '4']
#ss = ['5', '6', '7', '8']
#s.extend(ss)
#print(s)


#查看数据类型
#arr = np.array([1, 2, 3, 4, 5])
#print(arr.dtype)

#转换数据类型
#float_arr = arr.astype(np.float32)
#print(float_arr.dtype)

#double_arr = np.double(arr)
##print(type(double_arr)  # <class 'numpy.ndarray'>
#print(double_arr.dtype)

#字符串数组转换为数值型
#numeric_strings = np.array([1.2,'2.3','3.2141'], dtype=np.string_)
#print(numeric_strings)
#numeric_strings.astype(float)
#print(numeric_strings)


"flatten"
#flatten返回一个折叠成一维的数组。
#但是该函数只能适用于numpy对象，
#即array或者mat，普通的list列表是不行的。
#a = np.array([[1, 2], [3, 4]])
#print(a.flatten())
#aa = [[1, 2], [3, 4]]
#print([y for x in a for y in x])


"append and extend"
#music_media = ['compact disc', '8-track tape', 'long playing record']
#new_media = ['DVD Audio disc', 'Super Audio CD']
#music_media.append(new_media)
#print(music_media)
##['compact disc', '8-track tape', 'long playing record', ['DVD Audio disc', 'Super Audio CD']]

#music_media = ['compact disc', '8-track tape', 'long playing record']
#new_media = ['DVD Audio disc', 'Super Audio CD']
#
#music_media.extend(new_media)
#print(music_media)
#['compact disc', '8-track tape', 'long playing record', 'DVD Audio disc', 'Super Audio CD']

"set, .count()"
#mylist = [1,2,2,2,2,3,3,3,4,4,4,4]
#myset = set(mylist)
#for item in myset:
#    print("the %d has found %d" %(item,mylist.count(item)))
#    
##the 1 has found 1
##the 2 has found 4
##the 3 has found 3
##the 4 has found 4

"startswith"
#str.startswith(str, beg=0,end=len(string))
#str -- 检测的字符串。
#strbeg -- 可选参数用于设置字符串检测的起始位置。
#strend -- 可选参数用于设置字符串检测的结束位置。
#str = "this is string example....wow!!!"
#print(str.startswith( 'this' ))
#print(str.startswith( 'is', 2, 4 ))
#print(str.startswith( 'this', 2, 4 ))
#
##True
##True
##False


"np.uint8"
#data = np.array([-2, -6 , -2.7, -2.1, 9, 10, 0, -1])
#print(data.astype(np.uint8))



"json数据的导入与导出"
#import json
## save data to json file
#def store(data, file):
#    with open(file, 'w') as fw:
#        # 将字典转化为字符串
#        # json_str = json.dumps(data)
#        # fw.write(json_str)
#        # 上面两句等同于下面这句
#        json.dump(data,fw)
#        
#        
## load json data from file
#def load(file):
#    with open(file,'r') as f:
#        data = json.load(f)
#        return data
#
#
#if __name__ == "__main__":
#    json_data = '{"login":[{"username":"aa","password":"001"},{"username":"bb","password":"002"}],"register":[{"username":"cc","password":"003"},{"username":"dd","password":"004"}]}'
#    # loads函数是将json格式数据转换为字典
#    file = "write_data.json"
#    data = json.loads(json_data)
     #将字典转化为字符串
#    store(data)
#    # 
#    data = load(file)
#    print(data)
## 注：load, dump 用于操作文件， loads, dumps用于数据类型的转换


#def load(file):
#    with open(file,'r') as f:
#        data = json.load(f)
#        return data
#
#
#file = "tx_ts.json"
#data = load(file)
#print(type(data))
#data = data[0].values()
#data = np.array(list(data))
#print(data[0])


"npy数据的导入与导出"
# 存储一个数组
#a = np.random.rand(3, 3)
#np.save("data.npy", a)
#print(a)

# 存储多个数组
#a = np.random.rand(3, 3)
#b = np.random.rand(4, 4)
#
#np.savez("data.npz", a=a, b=b)

# 使用np.load()读取npz, npy文件
#data =  np.load("data.npz")
#print(data["a"])
#print(data["b"])

# 存储字典
#x = {0:'wpy', 1:'scg'}
#np.save("test.npy", x)
#x = np.load("test.npy")
##在存为字典格式读取后, 调用如下语句, 将数据np.array对象转换为dict
#x.item() 
#print(x)


"""
try...except
"""
# 程序不会因为异常而中断
#把可能发生错误的语句放在try模块里，
#用except来处理异常。except可以处理一个专门的异常，
#也可以处理一组圆括号中的异常，
#如果except后没有指定异常，则默认处理所有的异常。
#每一个try，都必须至少有一个except
#try后只有一个except会被执行
#try:
#    a = int(input("请输入除数:\n"))
#    b = int(input("请输入被除数:\n"))
#    c = a / b
#    print("您输入的两个数相除的结果是:\n", c )
#except IndexError:
#    print("索引错误：运行程序时输入的参数个数不够")
#except ValueError:
#    print("数值错误：程序只能接收整数参数")
#except ZeroDivisionError:
#    print("不能除以零！")
#
#except Exception:
#    pass
#
#print("Done")


## 多异常捕获
#try:
#    a = int(input("请输入除数:\n"))
#    b = int(input("请输入被除数:\n"))
#    c = a / b
#    print("您输入的两个数相除的结果是:\n", c )
#except(IndexError, ValueError, ArithmeticError):
#    print("程序发生了数组越界、数字格式异常、算术异常之一")
    

##访问异常信息
#def foo():
#    try:
#        open("a.txt");
#    except Exception as e:
#        # 访问异常的错误编号和详细信息
#        print(e.args)
#        # 访问异常的错误编号
#        print(e.errno)
#        # 访问异常的详细信息
#        print(e.strerror)
#foo()
#    
##(2, 'No such file or directory')
##2
##No such file or directory    
  
 
    
# else块
#def else_test():
#    s = input('请输入除数:')
#    result = 20 / int(s)
#    print('20除以%s的结果是: %g' % (s , result))
#
#def right_main():
#    try:
#        print('try块的代码，没有异常')
#    except:
#        print('程序出现异常')
#    else:
#        # 将else_test放在else块中
#        else_test()
#
#def wrong_main():
#    try:
#        print('try块的代码，没有异常')
#        # 将else_test放在try块代码的后面
#        else_test()
#    except:
#        print('程序出现异常')
#
#wrong_main()
#right_main()
#
##如果将 else_test() 函数放在 try 块的代码的后面，
##此时 else_test() 函数运行产生的异常将会被 try 块对应的 except 捕获，
##这正是 Python 异常处理机制的执行流程：
##但如果将 else_test() 函数放在 else 块中，
##当 else_test() 函数出现异常时，程序没有 except 块来处理该异常，
##该异常将会传播给 Python 解释器，导致程序中止。



"""
lambda
"""
#g = lambda x:x+1
#print(g(1))
##gg = lambda x:x+1(1)
##print(gg)
## python中定义好的全局函数: filter, map
#foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
#out_1 = list(filter(lambda x: x % 3 == 0, foo))
#out_2 = list(map(lambda x: x * 2 + 10, foo))
#
#print(out_1)
#print(out_2)



"""
lambda, filter
"""

#if __name__ == "__main__":
#    examine_scores = {"google": 98, "baidu": 95, "sougo": 90, "360": 80, 
#                      "yahoo": 90, "bing": 98, "QQ": 80, "IE": 85}
#    t = sorted(examine_scores.items(), key=lambda x: x[1], reverse=True)
#    print(t)
#    print("输出成绩表：")
#    for i in t:
#        print(i)
#    print("输出最高成绩：")
#    max_score = t[0][1]
#    print(list(filter(lambda x: x[1] == max_score, t)))
#    print("输出最低成绩：")
#    min_score = t[len(t) - 1][1]
#    print(list(filter(lambda x: x[1] == min_score, t)))
#    print("输出平均成绩：", sum(examine_scores.values()) / len(examine_scores.values())


#[('google', 98), ('bing', 98), ('baidu', 95), ('sougo', 90), 
# ('yahoo', 90), ('IE', 85), ('360', 80), ('QQ', 80)]
#输出成绩表：
#('google', 98)
#('bing', 98)
#('baidu', 95)
#('sougo', 90)
#('yahoo', 90)
#('IE', 85)
#('360', 80)
#('QQ', 80)
#输出最高成绩：
#[('google', 98), ('bing', 98)]
#输出最低成绩：
#[('360', 80), ('QQ', 80)]
#输出平均成绩： 89.5




"""
isinstance, type
"""
#isinstance() 与 type() 区别：
#type() 不会认为子类是一种父类类型，不考虑继承关系。
#isinstance() 会认为子类是一种父类类型，考虑继承关系。 
#如果要判断两个类型是否相同推荐使用 isinstance()。
# isinstance(object, classinfo)
#object -- 实例对象。
#classinfo -- 可以是直接或间接类名、基本类型
#或者由它们组成的元组。
#isinstance (a,(str,int,list))    # 是元组中的一个返回 True

#class A:
#    pass
# 
#class B(A):
#    pass
# 
#isinstance(A(), A)    # returns True
#type(A()) == A        # returns True
#isinstance(B(), A)    # returns True
#type(B()) == A        # returns False


"""
super
"""
#
## 单继承
#class A:
#    def __init__(self):
#        self.n = 2
#
#    def add(self, m):
#        print('self is {0} @A.add'.format(self))
#        self.n += m
#
#
#class B(A):
#    def __init__(self):
#        self.n = 3
#
#    def add(self, m):
#        print('self is {0} @B.add'.format(self))
#        super().add(m)
#        self.n += 3
#
#b = B()
#b.add(2)
#print(b.n)

#self is <__main__.B object at 0x000001E1BCA49208> @B.add
#self is <__main__.B object at 0x000001E1BCA49208> @A.add
#8



# 多继承



"zip"
#a = [1, 2, 3]
#b = [4, 5, 6]
#c = [4, 5, 6, 7, 8]
#d = ['one', 'two', 'three']
#zipped = list(zip(a, b))
#zipped_1 = list(zip(a, c))
#zipped_2 = list(zip(*zipped)) # 与zip相反， 可理解为解压
#zipped_3 = dict(zip(d, a))
#print(zipped)
#print(zipped_1)
#print(zipped_2)
#print(zipped_3)
##[(1, 4), (2, 5), (3, 6)]
##[(1, 4), (2, 5), (3, 6)]
##[(1, 2, 3), (4, 5, 6)]
##{'one': 1, 'two': 2, 'three': 3}

#string = ['a', 'b', 'c', 'd', 'e', 'f']
#zipped_string = list(zip(string[:-1], string[1:]))
#
#print(string)
#print(zipped_string)
#
##['a', 'b', 'c', 'd', 'e', 'f']
##[('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f')]
#
#nums = ['flower', 'flow', 'flight']
#for i in zip(*nums):
#    print(i)
#
##('f', 'f', 'f')
##('l', 'l', 'l')
##('o', 'o', 'i')
##('w', 'w', 'g')


   
    
    
"*, **"
## *parameter用来接收任意多个参数并将其放在
#
#def demo(*p):
#    print(p)
#   
#demo(1, 2, 3)
##(1, 2, 3)
#
#def d(a, b, c):
#    print(a, b, c)
#    
#a = [1, 2, 3]
#b = [4, 5, 6]
#c = [7, 8, 9]
#
#d(a, b, c)
##[1, 2, 3] [4, 5, 6] [7, 8, 9]
##函数在调用多个参数时，在列表、元组、集合、字典
##及其他可迭代对象作为实参，并在前面加 *
#d(*a)
##1 2 3
#
#def demo(**p):
#    for i in p.items():
#        print(i)

#demo(x=1, y=2)
#
##('x', 1)
##('y', 2)

#data= dict(rotation_range=0.2,
#            width_shift_range=0.05,
#            height_shift_range=0.05,
#            shear_range=0.05,
#            zoom_range=0.05,
#            horizontal_flip=True,
#            fill_mode='nearest')
#
#demo(**data)
#
#('rotation_range', 0.2)
#('width_shift_range', 0.05)
#('height_shift_range', 0.05)
#('shear_range', 0.05)
#('zoom_range', 0.05)
#('horizontal_flip', True)
#('fill_mode', 'nearest')


"继承类"

#class Father:
#    def __init__(self, handsome, tall):
#        self.handsome = handsome
#        self.tall = tall
#    
#class Son(Father):
#    def __init__(self, handsome, tall, weight):
#        super().__init__(handsome, tall)
#        self.weight = weight
#        
#    def run(self):
#        print('小明长得很%s, 个子%d, 体重%d'%(self.handsome, self.tall, self.weight))
#zhangsan = Son('吴彦祖', 180, 120)
#zhangsan.run()



"hasattr(), getattr(), setattr()"
#用于判断对象是否包含对应的属性
#hasattr(object, name)

#class Coordinate:
#    x = 10
#    y = -5
#    z = 0
#
#point = Coordinate()
#print(hasattr(point, 'x'))
#print(hasattr(point, 'y'))
#print(hasattr(point, 'z'))
#print(hasattr(point, 'no')) 
#
##True
##True
##True
##False

#class A():
#    name = 'python'
#    def func(self):
#        return 'Hello world'
#    
#print(hasattr(A, 'name'))
#print(hasattr(A, 'age'))
#print(hasattr(A, 'func'))
#
##True
##False
##True
# getattr(object, name[, default])
#print(getattr(A, 'name'))
#print(getattr(A, 'age', 20))
#print(getattr(A(), 'func')())
#
##python
##20
##Hello world




#class A():
#    name = 'python'
#    @classmethod
#    def func(self):
#        return 'Hello world'
#    
#print(getattr(A, 'func')())
##Hello world


# setattr(object, name, value)
#class A():
#    name = 'python'
#    def func(self):
#        return 'Hello world'
#
#setattr(A, 'name', 'java')
#print(getattr(A, 'name'))
#
#setattr(A, 'age', 20)
#print(getattr(A, 'age'))
#
##java
##20

"__dict__"
#class MainHandler:
#    def __init__(self):
#        self.host = "(.*)"
#        self.ip =  "^(25[0-5]|2[0-4]\d|[0-1]?\d?\d)(\.(25[0-5]|2[0-4]\d|[0-1]?\d?\d)){3}$"
#        self.port = "(\d+)"
#        self.phone =  "^1[3|4|5|8][0-9]\d{8}$"
#        self.file = "^(\w+\.pdf)|(\w+\.mp3)|(\w+\.py)$"
#
#    def chek(self):
#        for key,val in self.__dict__.items():
#            print(key,'-----',val)
#
#obj = MainHandler()
#
#obj.chek()
##host ----- (.*)
##ip ----- ^(25[0-5]|2[0-4]\d|[0-1]?\d?\d)(\.(25[0-5]|2[0-4]\d|[0-1]?\d?\d)){3}$
##port ----- (\d+)
##phone ----- ^1[3|4|5|8][0-9]\d{8}$
##file ----- ^(\w+\.pdf)|(\w+\.mp3)|(\w+\.py)$
#

"startswith()"
##str.startswith(substr, beg=0, end=len(string))
##str -- 检测的字符串。
##substr -- 指定的子字符串。
##beg -- 可选参数用于设置字符串检测的起始位置。
##end -- 可选参数用于设置字符串检测的结束位置。
#str = "this is string example....wow!!!"
#print (str.startswith( 'this' ))   # 字符串是否以 this 开头
#print (str.startswith( 'string', 8 ))  # 从第八个字符开始的字符串是否以 string 开头
#print (str.startswith( 'this', 2, 4 )) # 从第2个字符开始到第四个字符结束的字符串是否以 this 开头
##True
##True
##False



## coding:utf-8
#from __future__ import print_function
#import os
#from io import BytesIO
#import numpy as np
#from functools import partial
#import PIL.Image
#import scipy.misc
#import tensorflow as tf
#import cv2
#
#graph = tf.Graph()
#model_fn = 'tensorflow_inception_graph.pb'
##这是谷歌inception网络模型，已经在imagenet训练好
#sess = tf.InteractiveSession(graph=graph)
#with tf.gfile.FastGFile(model_fn, 'rb') as f:
#    graph_def = tf.GraphDef()
#    graph_def.ParseFromString(f.read())
#t_input = tf.placeholder(np.float32, name='input')
#imagenet_mean = 117.0
#t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
#tf.import_graph_def(graph_def, {'input': t_preprocessed})
#
#
#def savearray(img_array, img_name):
#    scipy.misc.toimage(img_array).save(img_name)
#    print('img saved: %s' % img_name)
#
#
#
#def resize_ratio(img, ratio):
#    min = img.min()
#    max = img.max()
#    img = (img - min) / (max - min) * 255
#    img = np.float32(scipy.misc.imresize(img, ratio))
#    img = img / 255 * (max - min) + min
#    return img
#
#
#def calc_grad_tiled(img, t_grad, tile_size=512):
#    sz = tile_size
#    h, w = img.shape[:2]
#    sx, sy = np.random.randint(sz, size=2)
#    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)  # 先在行上做整体移动，再在列上做整体移动
#    grad = np.zeros_like(img)
#    for y in range(0, max(h - sz // 2, sz), sz):
#        for x in range(0, max(w - sz // 2, sz), sz):
#            sub = img_shift[y:y + sz, x:x + sz]
#            g = sess.run(t_grad, {t_input: sub})
#            grad[y:y + sz, x:x + sz] = g
#    return np.roll(np.roll(grad, -sx, 1), -sy, 0)
#
#k = np.float32([1, 4, 6, 4, 1])
#k = np.outer(k, k)
#k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)
#
## 这个函数将图像分为低频和高频成分
#def lap_split(img):
#    with tf.name_scope('split'):
#        # 做过一次卷积相当于一次“平滑”，因此lo为低频成分
#        lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')
#        # 低频成分放缩到原始图像一样大小得到lo2，再用原始图像img减去lo2，就得到高频成分hi
#        lo2 = tf.nn.conv2d_transpose(lo, k5x5 * 4, tf.shape(img), [1, 2, 2, 1])
#        hi = img - lo2
#    return lo, hi
#
## 这个函数将图像img分成n层拉普拉斯金字塔
#def lap_split_n(img, n):
#    levels = []
#    for i in range(n):
#        # 调用lap_split将图像分为低频和高频部分
#        # 高频部分保存到levels中
#        # 低频部分再继续分解
#        img, hi = lap_split(img)
#        levels.append(hi)
#    levels.append(img)
#    return levels[::-1]
#
## 将拉普拉斯金字塔还原到原始图像
#def lap_merge(levels):
#    img = levels[0]
#    for hi in levels[1:]:
#        with tf.name_scope('merge'):
#            img = tf.nn.conv2d_transpose(img, k5x5 * 4, tf.shape(hi), [1, 2, 2, 1]) + hi
#    return img
#
#
## 对img做标准化。
#def normalize_std(img, eps=1e-10):
#    with tf.name_scope('normalize'):
#        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
#        return img / tf.maximum(std, eps)
#
## 拉普拉斯金字塔标准化
#def lap_normalize(img, scale_n=4):
#    img = tf.expand_dims(img, 0)
#    tlevels = lap_split_n(img, scale_n)
#    # 每一层都做一次normalize_std
#    tlevels = list(map(normalize_std, tlevels))
#    out = lap_merge(tlevels)
#    return out[0, :, :, :]
#
#
#def tffunc(*argtypes):
#    placeholders = list(map(tf.placeholder, argtypes))
#    def wrap(f):
#        out = f(*placeholders)
#        def wrapper(*args, **kw):
#            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
#        return wrapper
#    return wrap
#
#
#def render_lapnorm(t_obj, img0,
#                   iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
#    # 同样定义目标和梯度
#    t_score = tf.reduce_mean(t_obj)
#    t_grad = tf.gradients(t_score, t_input)[0]
#    # 将lap_normalize转换为正常函数
#    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))
#
#    img = img0.copy()
#    for octave in range(octave_n):
#        if octave > 0:
#            img = resize_ratio(img, octave_scale)
#        for i in range(iter_n):
#            g = calc_grad_tiled(img, t_grad)
#            # 唯一的区别在于我们使用lap_norm_func来标准化g！
#            g = lap_norm_func(g)
#            img += g * step
#            print('.', end=' ')
#    savearray(img, 'lapnorm.jpg')
#
#if __name__ == '__main__':
#    name = 'mixed4d_3x3_bottleneck_pre_relu'
#    channel = 79
#    #img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
#    #读取自己定义的图像，并设定大小
#    img_noise= cv2.imread("timg.jpg")
#    img_noise=cv2.resize(img_noise,(500,400))+100.0
#    layer_output = graph.get_tensor_by_name("import/%s:0" % name)
#    render_lapnorm(layer_output[:, :, :, channel], img_noise, iter_n=100)
#    #这里iter_n为迭代次数，越多效果越明显




#"load .mat"
#import scipy.io as sio
#import numpy as np
#
#
###load
##data = sio.loadmat('D:/Downloads/isbi_test810.mat')
##print(data)
#
###save
##array_x = np.array([1, 2, 3, 4])
##array_y = np.array([5, 6, 7, 8])
##sio.savemat('save.mat', {'arrayX': array_x, 'arrayY':array_y})
##
##data = sio.loadmat('save.mat')
##print(data)
#
##{'__header__': b'MATLAB 5.0 MAT-file Platform: nt, 
## Created on: Sat May  9 20:55:52 2020', 
##'__version__': '1.0',  '__globals__': [], 
## 'arrayX': array([[1, 2, 3, 4]]), 'arrayY': array([[5, 6, 7, 8]])}

#v7.3版本的.mat文件是matlab中保存大文件的格式，使用上面的方式是无法读取的，这个时候需要使用h5py
#安装h5py:http://blog.csdn.net/GYGuo95/article/details/79537594
#
#import h5py
#data = h5py.File('data.mat')




"""
plot a confusion matrix
"""
# =============================================================================
# #import numpy as np
# 
# 
# def plot_confusion_matrix(cm,
#                           target_names,
#                           title='Confusion matrix',
#                           cmap=None,
#                           normalize=True):
#     """
#     given a sklearn confusion matrix (cm), make a nice plot
# 
#     Arguments
#     ---------
#     cm:           confusion matrix from sklearn.metrics.confusion_matrix
# 
#     target_names: given classification classes such as [0, 1, 2]
#                   the class names, for example: ['high', 'medium', 'low']
# 
#     title:        the text to display at the top of the matrix
# 
#     cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
#                   see http://matplotlib.org/examples/color/colormaps_reference.html
#                   plt.get_cmap('jet') or plt.cm.Blues
# 
#     normalize:    If False, plot the raw numbers
#                   If True, plot the proportions
# 
#     Usage
#     -----
#     plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
#                                                               # sklearn.metrics.confusion_matrix
#                           normalize    = True,                # show proportions
#                           target_names = y_labels_vals,       # list of names of the classes
#                           title        = best_estimator_name) # title of graph
# 
#     Citiation
#     ---------
#     http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# 
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import itertools
# 
#     accuracy = np.trace(cm) / float(np.sum(cm))
#     misclass = 1 - accuracy
# 
#     if cmap is None:
#         cmap = plt.get_cmap('Blues')
# 
#     plt.figure(figsize=(8, 6))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
# 
#     if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names, rotation=45)
#         plt.yticks(tick_marks, target_names)
# 
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# 
# 
#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:0.4f}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
# 
# 
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
#     plt.show()
# 
# 
# #plot_confusion_matrix(cm           = np.array([[ 1098,  1934,   807],
# #                                              [  604,  4392,  6233],
# #                                              [  162,  2362, 31760]]), 
# #                      normalize    = False,
# #                      target_names = ['high', 'medium', 'low'],
# #                      title        = "Confusion Matrix")
# 
#     
# #plot_confusion_matrix(cm           = np.array([[ 1098,  1934,   807],
# #                                              [  604,  4392,  6233],
# #                                              [  162,  2362, 31760]]), 
# #                      normalize    = True,
# #                      target_names = ['high', 'medium', 'low'],
# #                      title        = "Confusion Matrix, Normalized")
# 
# =============================================================================



"嵌套列表，将其中同位置的元素组成新的列表"
##原文链接：https://blog.csdn.net/littlle_yan/article/details/81018149
#lsts = [[1,2,3], [4,5,6],[7,8,9],[10,11,12]] 
#ret_x = [x for [x,y,z] in lsts]
#ret_y = [y for [x,y,z] in lsts]
#ret_z = [z for [x,y,z] in lsts] 
#print(ret_x)  #输出结果[1, 4, 7, 10]
#print(ret_y)  #输出结果[2, 5, 8, 11]
#print(ret_z)  #输出结果[3, 6, 9, 12]







































