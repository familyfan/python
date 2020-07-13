# -*- coding: utf-8 -*-
import numpy as np  # numeric python 
import random 
import matplotlib.pyplot as plt
"参考网址:"
# https://blog.csdn.net/zz2230633069/article/details/81363630


#求矩阵的范数示例，ord=2求的是2范数
#norm = np.linalg.norm(A[:, 0], ord=2)

#array = np.arange(12).reshape(3, 4)
#print("Original array : \n", array)
#print("\nRolling with 1 shift : \n", np.roll(array, 1)) 
#print("\nRolling with 5 shift : \n", np.roll(array, 5)) 
##axis = 0 表示按行
#print("\nRolling with 5 shift with 0 axis : \n", np.roll(array, 2, axis = 0))
#print("\nRolling with 5 shift with 0 axis : \n", np.roll(array, 1, axis = 0))
#
#print("\nRolling with 5 shift with 0 axis : \n", np.roll(array, 1, axis = 1))
#print("\nRolling with 5 shift with 0 axis : \n", np.roll(array, 2, axis = 1))

# =============================================================================
# A = np.arange(1, 51).reshape(-1)
# print(A.shape)
# 
# B = nr.randint(1, 9, (10, 10))
# print(B)
# print(type(B))
# print(B[2])
# =============================================================================

#train = [[1, np.nan, 32, 1, 5, 12],
#         [2, 3, np.nan, 3, 4, 5]]
#where_nan = np.isnan(train)
##train[where_nan] = 0

"""
矩阵运算
"""
#A = np.matrix([[1, 2],
#               [3, 4]])
#print(A.I)
#print(A.T)
#print(np.linalg.det(A))

"""
数组运算
"""
# =============================================================================
# A = np.arange(1, 17).reshape(4,4)
# print(A)
# print(A[-1,-1])
# print("\n")
# print(A[2:,2])
# a = np.argmax(A[2:,2])
# print(a)
# print("\n")
# print(A[[2,3],1])
# print("\n")
# 
# print(np.tril(A,0))
# print("\n")
# print(np.tril(A,-1))
# print("\n")
# print(np.triu(A,0))
# =============================================================================
#"np.concatenate"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
c = np.concatenate((a, b), axis=0)
d = np.concatenate((a, b)) #默认为axis=0
print(c)
print(d)
print("\n")
#c1 = np.concatenate((a, b.T), axis=1)
#print(c1)
#print(b.shape)
#print(len(b))
#print(b.T.shape)



# 类似于seed, 得到的两个数相等
#state = np.random.get_state()
#chance = np.random.randint(100)
#np.random.set_state(state)
#chance2 = np.random.randint(100)
# 
#print(chance,chance2)


#arr = np.array(range(9)).reshape(3, 3)
#print(arr)
#arr1 = arr.repeat(3)
#print(arr1)
#arr2 = np.tile(arr, 2)
#print(arr2)
#[[0 1 2]
# [3 4 5]
# [6 7 8]]
#
#[0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8]
#
#[[0 1 2 0 1 2]
# [3 4 5 3 4 5]
# [6 7 8 6 7 8]]

#print(np.repeat(7, 4))
#[7 7 7 7]
#arr3 = np.array([10, 20])
#print(arr3.repeat([3, 2]))
#[10 10 10 20 20]
#print(np.repeat(arr3, [3, 2]))
#[10 10 10 20 20]
#a=np.array([[10,20],[30,40]]) 
#print(a.repeat([3,2],axis=0))
#[[10 20]
# [10 20]
# [10 20]
# [30 40]
# [30 40]]
#print(a.repeat([3,2],axis=1))
#[[10 10 10 20 20]
# [30 30 30 40 40]]

#def square(x):
#    return x **2
#print(map(square, [1, 2, 3, 4, 5]))
#print(map(lambda x: x**2, [1, 2, 3, 4, 5]))
#print(map(lambda x, y: x+y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10]))

#<map object at 0x000001B15565F1D0>
#<map object at 0x000001B15565F1D0>
#<map object at 0x000001B15565F208>

#list = [20, 16, 10, 5]
#random.shuffle(list)
#print("随机排序列表:", list)


# 以相同的方式随机打乱images和labels
#state = np.random.get_state()
#np.random.shuffle(training_images)
#np.random.set_state(state)
#np.random.shuffle(training_labels)



#np.asarray 与 np.array
#
#np.save(OUTPUT_FILE, processed_data)


"np.argsort"
#x = np.array([3, -5, 2, 4, 6])
#print(np.argsort(x))  # 按升序排列
#print(np.argsort(-x)) # 按降序排列
#print(x[np.argsort(x)])
#print(x[np.argsort(-x)])
#print(x[::-1])
#[1 2 0 3 4]
#[4 3 0 2 1]
#[-5  2  3  4  6]
#[ 6  4  3  2 -5]
#[ 6  4  2 -5  3]

#x = np.array(range(1, 10)).reshape(3, 3)
#print(x.get_shape().as_list())
#print(x)
#print(x[::-1])
#[[1 2 3]
# [4 5 6]
# [7 8 9]]
#
#[[7 8 9]
# [4 5 6]
# [1 2 3]]


"np.maximum, np.max"
#np.max(a, axis=None, out=None, keepdims=False)
#np.maximum(X, Y, out=None)
#print(np.maximum([-2, -1, 0, 1, 2], 0))    # [0 0 0 1 2]
#print(np.max([-2, -1, 0, 1, 2]))          # 2


#a = [-3, -2, -1, 0, 1, 2, 3]
#b = np.max(a)
#c = np.argmax(a)
#d = np.maximum(0, a)
#print(a)
#print(b)
#print(c)
#print(d)
##[-3, -2, -1, 0, 1, 2, 3]
##3
##6
##[0 0 0 0 1 2 3]
 
#print(np.maximum([2, 3, 4], [1, 5, 2]))  # [2 5 4]

"np.around, np.floor, np.ceil"
#array = np.array([-0.746, 4.6, 9.4, 7.447, 10.455, 11.555])
#print(np.around(array)) # 四舍五入
#print(np.floor(array)) # 向下取整
#print(np.ceil(array))  # 向上取整

#[-1.  5.  9.  7. 10. 12.]
#[-1.  4.  9.  7. 10. 11.]
#[-0.  5. 10.  8. 11. 12.]


"np.where"
#np.where(condition, x, y)
# 满足条件输出x, 不满足输出y.
#x = np.arange(10)
#print(x)
#print(np.where(x > 5, 1, 0))

#[0 1 2 3 4 5 6 7 8 9]
#[0 0 0 0 0 0 1 1 1 1]

# 只有条件, 没有x和y则输出满足条件元素的坐标
#坐标以tuple的形式给出，通常原数组有多少维，
#输出的tuple中就包含几个数组，
#分别对应符合条件元素的各维坐标。
#a = np.array([2, 4, 6, 8, 10])
#print(np.where(a>5))
##(array([2, 3, 4], dtype=int64),)
#print(a[np.where(a>5)]) # 等价于a[a>5]
##(array([2, 3, 4], dtype=int64),)
#print(np.where([[0, 1], [1, 0]]))
##(array([0, 1], dtype=int64), 
## array([1, 0], dtype=int64))
#b = np.arange(27).reshape(3, 3, 3)
##print(b)
##print(b[b>5])
##print(np.where(b>5))

#[[[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]]
#
# [[ 9 10 11]
#  [12 13 14]
#  [15 16 17]]
#
# [[18 19 20]
#  [21 22 23]
#  [24 25 26]]]
#[ 6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26]
#(array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
#       dtype=int64), 
# array([2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2],
#       dtype=int64), 
# array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
#       dtype=int64))

"np.random.shuffle()"

#arr= np.arange(9).reshape(3, 3)
#print(arr)
#arr = np.random.shuffle(arr)
#print(arr)

"np.load"
#各个格式文件的存取：
#np格式：np.save,  np.load
#znp格式：np.savez,  np.load
#csv文件：np.savetxt,  np.loadtxt

"convert"
#from numpy import *
#a1 =[1,2,3,4,5,6] #列表
#print('a1 :',a1)
##('a1 :', [1, 2, 3, 4, 5, 6])
# 
#a2 = array(a1)   #列表 -----> 数组
#print('a2 :',a2)
##('a2 :', array([1, 2, 3, 4, 5, 6]))
# 
#a3 = mat(a1)      #列表 ----> 矩阵
#print('a3 :',a3)
##('a3 :', matrix([[1, 2, 3, 4, 5, 6]]))
# 
#a4 = a3.tolist()   #矩阵 ---> 列表
#print('a4 :',a4)
##('a4 :', [[1, 2, 3, 4, 5, 6]])   #注意！！有不同
#print(a1 == a4)
##False
# 
#a8 = a3.tolist()[0]   #矩阵 ---> 列表
#print('a8 :',a8)
##('a8 :', [1, 2, 3, 4, 5, 6])  #注意！！有不同
#print(a1 == a8)
##True
# 
#a5 = a2.tolist()   #数组 ---> 列表
#print('a5 :',a5)
##('a5 :', [1, 2, 3, 4, 5, 6])
#print(a5 == a1)
##True
# 
#a6 = mat(a2)   #数组 ---> 矩阵
#print('a6 :',a6)
##('a6 :', matrix([[1, 2, 3, 4, 5, 6]]))
# 
#print(a6 == a3)
##[[ True  True  True  True  True  True]]
# 
#a7 = array(a3)  #矩阵 ---> 数组
#print('a7 :',a7)
##('a7 :', array([[1, 2, 3, 4, 5, 6]]))
#print(a7 == a2)
##[[ True  True  True  True  True  True]]
#

"np.meshgrid"
#nx, ny = (4, 5)
#x = np.linspace(0, 1, nx)
#y = np.linspace(0, 1, ny)
#xv, yv = np.meshgrid(x, y)
#print(x)
#print(y)
#print(xv)
#print(yv)
##[0.         0.33333333 0.66666667 1.        ]
##[0.   0.25 0.5  0.75 1.  ]
##[[0.         0.33333333 0.66666667 1.        ]
## [0.         0.33333333 0.66666667 1.        ]
## [0.         0.33333333 0.66666667 1.        ]
## [0.         0.33333333 0.66666667 1.        ]
## [0.         0.33333333 0.66666667 1.        ]]
##[[0.   0.   0.   0.  ]
## [0.25 0.25 0.25 0.25]
## [0.5  0.5  0.5  0.5 ]
## [0.75 0.75 0.75 0.75]
## [1.   1.   1.   1.  ]]

#nx, ny = (3, 2)
#x = np.linspace(0, 1, nx)
#y = np.linspace(0, 1, ny)
#xv, yv = np.meshgrid(x, y, sparse=True)  # make sparse output arrays
#print(xv)
#print(yv)
##[[0.  0.5 1. ]]
##
##[[0.]
## [1.]]


#x = np.arange(-5, 5, 0.1)  # x in -5+0.1*h
#y = np.arange(-5, 5, 0.1)
#xx, yy = np.meshgrid(x, y, sparse=True)
#z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
#h = plt.contourf(x,y,z)
#plt.show()
##zz = np.sin(xx**2 + yy**2)
##hh = plt.contourf(x,y,zz)
##plt.show()

"np.random.randn"
#x = np.random.randn(3, 2)
#print(x)

"np.c_, np.r_"
#a = np.array([[1, 2, 3],[7,8,9]])
#b = np.array([[4,5,6],[1,2,3]])
#c = np.c_[a, b]
#d = np.r_[a, b]
#print(c)
#print(d)
##[[1 2 3 4 5 6]
## [7 8 9 1 2 3]]

##[[1 2 3]
## [7 8 9]
## [4 5 6]
## [1 2 3]]

#X = np.random.randn(50, 2)
#X_train = np.r_[X, X]
#print(X)
#print("\n")
#print(X_train)


#X = 0.3 * np.random.randn(2, 2, 3)
#print(X)
#X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
#print(X_outliers)


X = np.random.randn(2, 3, 4)
print(X)
print(X.reshape(6, 4))
















"size() 计算数组和矩阵所有数据的个数"
#a = np.array([[2, 4, 6], [1, 3, 5]])
#print(a.size)
#print(np.size(a))
#print(np.size(a, 0))
#print(np.size(a, 1))
##6
##6
##2
##3
#a = np.array([[2, 4, 6], [1, 3, 5]])
#print(a[5>4].size)
#print(a[0].size)
#print(a[1].size)
#print(a[0], a[1])
#print(a[5>4])
#print(a[5==4])
#print(a[5<4])
##6
##3
##3
##[2 4 6] [1 3 5]
##[[[2 4 6]
##  [1 3 5]]]
##[]
##[]






























































































































