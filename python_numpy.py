# -*- coding: utf-8 -*-
#import numpy
#print(numpy.__version__)  #查看NumPy的版本号
#创建数组
import numpy as np
#a = np.array([1, 2, 3, 4]) 
#b = np.array((5, 6, 7, 8))
#c = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
# =============================================================================
# print(a.shape)
# print(b.shape)
# print(c.shape)
# c.shape = 4, 3  #也可以加括号写成 (4, 3)
# print(c)
# #当设置某个轴的元素个数为-1时，将自动计兑此轴的长度
# #将数组c的shape属性改成了(2,6)
# c.shape = 2, -1
# print(c)
# #使用数组的reshapeO方法，可以创建指定形状的新数组，而原数组的形状保持不变
# d = a.reshape((2, 2))  #也可以写成 a.reshape(2, 2)
# print(d)
# print(a)
# #数组a和d其实共享数椐存储空间例如：
# a[1] = 100
# print(a)
# print(d)
# 
# #元素类型
# print(c.dtype)
# #可以通过dtype参数在创建数组吋指定元素类型，
# #注意float类型是64位的双精度浮点类型，
# #而complex是128位的双精度复数类型：
# ai32 = np.array([1, 2, 3, 4], dtype=np.int32)
# af = np.array([1, 2, 3, 4], dtype=float) 
# ac = np.array([1, 2, 3, 4], dtype=complex)
# print(ai32.dtype)
# print(af.dtype)
# print(ac.dtype)
# =============================================================================

# =============================================================================
# #完整的类型列表
# print(set(np.typeDict.values()))
# #通过dtype对象的type属性可以获 得与其对应的数值类型：
# print(c.dtype.type)
# #通过NumPy的数值类型也可以创建数值对象，下面创建一个16位的符号整数对象
# a = np.int16(200)
# print(a*a)  #取值范围有限，导致溢出
# #NumPy的数值对象的运兑速度比Python的内置类型的运兑速度慢很多
# #直接在右边console中实现
# #vl = 3.14
# #v2 = np.float64(vl)
# #%timeit vl*vl
# #%timeit v2*v2
# #得到如下结果：
# #42.4 ns ± 0.651 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
# #71.5 ns ± 0.519 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
# #使用astypeO方法可以对数组的元素类型进行转换
# tl = np.array([1, 2, 3, 4], dtype=np.float) 
# t2 = np.array([1, 2, 3, 4], dtype=np.complex) 
# t3 = tl.astype(np.int32) 
# t4 = t2.astype(np.complex64)
# 
# #自动生成数组
# #通过指定开始值、终值和步长来创建表示等差数列的一维数组,但终值不在数组中
# print(np.arange(0, 1, 0.1))
# #指定幵始值、终值和元素个数来创建表示等差数列的一维数组
# print(np.linspace(0, 1, 10))  #默认包含终值，步长为1/9
# print(np.linspace(0, 1, 10, endpoint=False))  #步长为1/10
# #产生从到10^0到10^2,有5个元素的等比数列，
# #注意起始值0表示10^0,而终值2表示10^2
# print(np.logspace(0, 2, 5))
# #基数可以通过base参数指定，其默认值为10
# print(np.logspace(0, 1, 12, base=2, endpoint=False))
# #zeros()、ones()、empty()可以创建指定形状和类型的数组
# print(np.empty((2, 3), np.int))
# print(np.zeros(4, np.int))
# print(np.ones(4, np.int))
# #full()将数组元素初始化为指定的值
# print(np.full(4, np.pi))
# #zero_like(),ones_like(),empty_like,full_like()
# #等函数创建与参数数组的形状和类型相同
# #例如：zerosjike ( a )和 zeros ( a . shape ， a . dtype )的效果相同
# =============================================================================

# =============================================================================
# #frombuffer(),iVomstring(),fromfile()等函数可以从字节序列或文件创建数组
# #以fromstring()为例介绍用法
# s = "abcdefgh"
# #Python 的字符串实际上是一个字节序列，每个字符占一个字节
# print(np.fromstring(s, dtype=np.int8))
# #字节98和字节97当作一个16位的整数，它的值就是98*256+97 = 25185
# print(np.fromstring(s, dtype=np.int16))  
# #fromstring()会对字符串的字节序列进行复制，而使用 frombuffer()创建的数组与原始字符串
# #共享内存.由于字符串萣只读的，因此无法修改所创建的数组的内容.
# 
# #定义一个从下标计算数值的函数，然后用 fromfunction()通过此函数创建数组
# def func(i):
#     return i % 4 + 1
# 
# print(np.fromfunction(func, (10,)))
# #创建一个表示九九乘法表的二维数组，输出的数组a中的每个元素 a[i,j]都等于func2(i,j )
# def func2(i, j):
#     return (i + 1) * (j + 1)
# print(np.fromfunction(func2, (9,9)))
# 
# #存取元素
# a = np.arange(10)
# print(a)
# print(a[5])
# print(a[3:5])
# print(a[:5])
# print(a[1:-1:2]) #2表示隔一个元素取一个元素
# print(a[::-1])  #整个数组头尾颠倒
# print(a[5:1:-2])  #步长为负数时，开始下标必须大于结束下标
# #用下标修改元素值
# a[2:4] = 100, 101
# print(a)
# #切片获取的新的数组是原始数组的一个视图,
# #它与原始数组共享同一块数椐存储空间
# #将 b 的第2 个元素修改为-10, a 的第5 个元素也同时被修改为-10
# b = a[3:7]
# b[2] = -10
# print(b)
# print(a)
# =============================================================================
# =============================================================================
# #当使用整数列表对数组元素进行存取时，将使用列表中的每个元素作为下标
# #使用列表作为下标得到的数组不和原始数组共享数据
# x = np.arange(10, 1, -1)
# print(x)
# a = x[[3, 3, 1, 8]]
# b = x[[3, 3, -3, 8]]
# print(a)
# print(b)
# b[2] = 100
# print(b)
# print(x)  #由于b不和x共享内存，因此x的值不变
# x[[3, 5, 1]] = -1, -2, -3
# print(x)  #整数序列下标也可以用来修改元素的值
# print(x[np.array([3,3,1,8])])  #输出[7,7,9,2]
# print(x[np.array([[3,3,1,8],[3,3,-3,8]])])  #同样输出多维数组
# print(x[[3,3,1,8,3,3,-3,8]].reshape(2,4))  #改变数组形状
# #使用布尔数组 b 作为下标存取数组 x 中的元素时，
# #将获得数组 x 中与数组 b 中 True 对应的元素
# #使用布尔数组作为下标获得的数组不和原始数组共享数据内存
# x = np.arange(5, 0, -1)
# print(x)
# print(x[np.array([True, False, True, False, False])])
# #在NumPy1.10之后的版本布尔列表会被当作布尔数组
# print(x[[True, False, True, False, False]])
# ##布尔数组的长度不够时，不够的部分都当作 False
# #print(x[np.array([True, False, True, True])])  #在此版本会报错
# x[np.array([True, False, True, True,False])] = -1, -2, -3
# print(x)
# #产生一个长度为 6 ，元素值为 0 到 9 的随机幣数数姐
# x = np.random.randint(0, 10, 6)
# print(x)
# print(x>5)
# print(x[x>5])  #得到x中所有大于5的数值所组成的布尔数组
# =============================================================================

# =============================================================================
# #多维数组
# #NumPy 采用元组作为数组的下标
# #a[1,2]和a[(1,2)]完全相同，都是使用元组(1,2)作为数组 a 的下标
# a = np.arange(0, 60, 10).reshape(-1, 1) + np.arange(0, 6)
# print(a)
# print(a[0, 3:5])  #注意下标从零开始
# print(a[4:, 4:])
# print(a[:, 2])
# print(a[2::2, ::2])  #从第二行开始间隔一个元素取值，再间隔一行取值
# print(a[2::2, ::])   #从第二行开始逐个元素取值，再间隔一行取值
# #如果 K 标元组中只包含整数和切片，那么得到的数组和原始数组共享数据
# b = a[0, 3:5]
# b[0] = -b[0]
# print(a[0, 3:5])
# #将下标元组保存起来，用同一个元组存取多个数组
# # a[idx]和 a[::2, 2:]相同，a[idx][idx]和a[::2, 2:][::2, 2:]相同
# idx = slice(None, None, 2), slice(2, None)
# print(a[idx])
# #用s_对象来帮助我们创建数组下标
# print(np.s_[::2, 2:])
# #在多维数组的下标元组中，也可以使用整数元组或列表、整数数组和布尔数组
# #当下标中使用这些对象时，所获得的数椐是原始数据的副木
# print(a[(0,1,2,3),(1,2,3,4)])  #分别取行和列，对应a[0,1],a[1,2],a[2,3],a[3,4]
# print(a[3:, [0,2,5]])
# #在 a [ mask , 2]中，第 0 轴的下标是一个布尔数组，它选取第0、第 2和第5 行；
# #第 1轴的下标是一个整数，它选収第2 列
# mask = np.array([1,0,1,0,0,1], dtype=np.bool)
# print(a[mask,2])
# #注意，如果 mask 不是布尔数组而是整数数组、列表或元组，
# #就按照以整数数组作为下标的方式进行运算
# mask1 = np.array([1,0,1,0,0,1])
# mask2 = [True,False,True,False,False,True]
# print(a[mask1, 2])
# print(a[mask2, 2])
# #下标的长度小于数组的维数时，剩余的各轴所对应的下标是":"
# print(a[[1,2],:])
# print(a[[1,2]])
# #当所有轴都用形状相冋的整数数组作为下标时，得到的数组和 K 标数组的形状相同
# x = np.array([[0,1],[2,3]])
# y = np.array([[-1,-2],[-3,-4]])
# print(a[x,y])  
# #结果与下面的程序相同：
# print(a[(0,1,2,3),(-1,-2,-3,-4)].reshape(2,2))
# print(np.array([[5, 14],
#              [23, 32]]))
# =============================================================================

#palette = np.array( [ [0, 0, 0],
#                      [255, 0, 0],
#                      [0, 255, 0],
#                      [0, 0, 255],
#                      [255, 255, 255] ])
#
#image = np.array( [ [ 0, 1, 2, 0 ],
#                    [ 0, 3, 4, 0 ] ] )
#print(palette[image])
#
###结构数组
#
#persontype = np.dtype({
#         'names':['name', 'age', 'weight'],
#        'formats':['S30', 'i', 'f']}, align=True)
#a = np.array([("Zhang", 32, 75.5), ("Wang", 24, 65.2)],
#              dtype=persontype)
#print(a.dtype)
#
##'S30’：长度为30个字节的字符串类型，由于结构中的每个元素的大小必须同定，
##因此需要指定字符串的长度
##'i':32位的整数类型，相当于np.int32
##'f':32位的单精度浮点数类型，相当于np.float32


# =============================================================================
# #2.2 ufunc(universal function)函数,一种能对数组的每个元素进行运算的函数
# x = np.linspace(0, 2*np.pi, 10)
# y = np.sin(x)
# print(y)
# #通过out参数指定保存计算结果的数组
# t = np.sin(x, out=x)
# print(t is x)
# #使用item()获取数组中的单个元素，并且直接返冋标准的Python数值类型
# a = np.arange(6.0).reshape(2,3)
# print(a.item(1, 2), type(a.item(1, 2)), type(a[1,2]))
# #四则运算
# a = np.arange(0, 4)
# b = np.arange(1, 5)
# print(np.add(a, b))  #求和
# print(np.add(a, b, a)) #保存指定数组a
# print(3.0//2)  #对返回值取整
# print(np.array([1,2,3]) < np.array([3,2,1]))
# a = np.arange(5)
# b = np.arange(4, -1, -1)
# print(np.logical_or(a == b, a>b))  #对数组进行"或运算"
# #以bitwisejf头的函数是位运算函数，
# #包括 bitwise_and、bitwise_not、bitwise_or 和 bitwise_xor等
# #也可以使用&、〜、丨和^等操作符进行计算
# ~ np.arange(5) #对正数按位取反将得到负数,以零为例，在32位符号整数中表示-1
# #自定义ufunc函数
# #用frompyftinc()将计算单个元素的函数转换成ufunc函数
# def triangle_wave(x, c, c0, hc):
#     x = x - int(x) #三角波的周期为1，因此只取x坐标的小数部分进行计算
#     if x >= c:
#         r = 0.0
#     elif x < c0:
#         r = x / c0 * hc
#     else:
#         r = (c - x) / (c - c0) * hc
#     return(r)
# #frompyfunc(func, nin, nout)
# #func是计算单个元素的函数，nin是func的输入参数的个数
# #nout是func的返回值的个数
# triangle_ufunc1 = np.frompyfunc(triangle_wave, 4, 1)
# y2 = triangle_ufunc1(x, 0.6, 0.4, 1.0)
# print(y2.dtype) #返回数组的元素类型是object
# print(y2.astype(np.float).dtype) #将其转换为双精度浮点数组
# #使用vectorize()也可以实现和frompyfunc()类似的功能
# #但它可以通过otypes参数指定返回的数组的元素类型
# triangle_ufunc2 = np.vectorize(triangle_wave, otypes=[np.float])
# y3 = triangle_ufunc2(x, 0.6, 0.4, 1.0)
# print(np.all(y2==y3))
# #广播(broadcasting)
# a  = np.arange(0, 60, 10).reshape(-1, 1)
# print(a, a.shape)
# b = np.arange(0, 5)
# print(b, b.shape)
# c = a + b
# print(c, c.shape)
# #为节省内存空间,NumPy提供了 ogrid对象,用于创建广播运算用的数组
# x, y = np.ogrid[:5, :5]
# print(x)
# print(y)
# #mgrid对象返回的是广播之后的数组：
# x, y = np.mgrid[:5, :5]
# print(x)
# print(y)
# x, y = np.ogrid[:1:4j, :1:3j]
# print(x)
# print(y)
# a = np.arange(4)
# print(a[None, :])
# print(a[:,None])
# print('\n')
# x = np.array([0, 1, 4, 10]) 
# y = np.array([2, 3, 8])
# print(x)
# print(y)
# print(x[None, :] + y[:, None])
# print('\n')
# #以使用ix_()将两个一维数组转换成可广播的二维数组
# gy, gx = np.ix_(y, x)
# print(gx)
# print(gy)
# print(gx + gy)
# =============================================================================


# =============================================================================
# #ufunc的方法
# r1 = np.add.reduce([1, 2, 3])
# r2 = np.add.reduce([[1, 2, 3], [4, 5, 6]], axis=1)
# print(r1)
# print(r2)
# print('\n')
# a1 = np.add.accumulate([1, 2, 3])
# a2 = np.add.accumulate([[1, 2, 3], [4, 5, 6]], axis=1)
# print(a1)
# print(a2)
# a = np.array([1, 2, 3, 4])
# result = np.add.reduceat(a, indices=[0, 1, 0, 2, 0, 3, 0]) 
# print(result)
# #outter()方法,乘法表最终是通过广播的方式计算出来的
# print(np.multiply.outer([1, 2, 3, 4, 5], [2, 3, 4]))
# #多维数组的下标存取
# #下标对象
# #为了避免出现问题，请“显式”地使用元组作为下标
# a = np.arange(3 * 4 * 5).reshape(3, 4, 5)
# lidx = [[0], [1]]
# aidx = np.array(lidx)
# print(a[lidx])
# print(a[aidx])
# i0 = np.array([[1, 2, 1], [0, 1, 0]])
# i1 = np.array([[[0]], [[1]]])
# i2 = np.array([[[2, 3, 2]]])
# b = a[i0, i1, i2]
# print(b)
# ind0, ind1, ind2 = np.broadcast_arrays(i0, i1, i2)
# print(ind0)
# print(ind1)
# print(ind2)
# 
# =============================================================================

# =============================================================================
# #庞大的函数库
# #随机数
# from numpy import random as nr
# np.set_printoptions(precision=2) #显示小数点后两位
# r1 = nr.rand(4, 3)
# r2 = nr.randn(4, 3) #标准正太分布的随机数
# r3 = nr.randint(0, 10, (4, 3))
# print(r1)
# print(r2)
# print(r3)
# print('\n')
# r1 = nr.normal(100., 10, (4, 3)) #正态分布
# r2 = nr.uniform(10, 20, (4, 3)) #均匀分布
# r3 = nr.poisson(2.0, (4, 3))  
# print(r1)
# print(r2)
# print(r3)
# print('\n')
# #permutation()用于产生一个乱序数组
# a = np.array([1, 10, 20, 30, 40])
# print(nr.permutation(10))
# print(nr.permutation(a))
# #shuffle()则直接将参数数组的顺序打乱
# nr.shuffle(a)
# print(a)
# print('\n')
# a = np.arange(10, 25, dtype=float)
# c1 = nr.choice(a, size=(4, 3))
# c2 = nr.choice(a, size=(4, 3), replace=False) #不重复抽取
# c3 = nr.choice(a, size=(4, 3), p=a / np.sum(a)) #指定每个元素对应的抽取概率
# print(c1)
# print(c2)
# print(c3)
# print('\n')
# =============================================================================
# =============================================================================
# #为了保证每次运行时能重现相同的随机数，可以通过seed〇函数指定随机数的种子
# #r3和r4得到的随机数组是相同的
# from numpy import random as nr
# r1 = nr.randint(0, 100, 3) 
# r2 = nr.randint(0, 100, 3) 
# nr.seed(42)
# r3 = nr.randint(0, 100, 3) 
# nr.seed(42)
# r4 = nr.randint(0, 100, 3)
# print(r1, r2)
# print(r3, r4)
# print('\n')
# #求和，平均值，方差
# #mean()表示求期望，std()表示标准差，product()表示连乘积
# np.random.seed(42)
# a = np.random.randint(0, 10,size=(4,5))
# print(a)
# print(np.sum(a))
# #保持原数组的维数
# print(np.sum(a, 1, keepdims=True))
# print(np.sum(a, 0, keepdims=True))
# print('\n')
# print(np.sum(a, axis=1)) #按行计算
# print(np.sum(a, axis=0)) #按列计算
# print(np.ones((2, 3, 4)))
# print(np.sum(np.ones((2, 3, 4)), axis=(0, 2)))
# np.set_printoptions(precision=8)
# b = np.full(1000000, 1.1, dtype=np.float32)
# print(b)
# #整数数组使用双精度浮点数进行计算
# print(np.mean(a, axis=1))
# print(np.mean(b))
# print(np.mean(b, dtype=np.double))
# #average()也可以对数组进行平均计算
# score = np.array([83, 72, 79]) 
# number = np.array([20, 15, 30]) 
# print(np.average(score, weights=number)) #指定权重
# print('\n')
# a = nr.normal(0, 2.0, (100000, 10))
# v1 = np.var(a, axis=1, ddof=0) #可以省略 ddof=0,默认为无偏样本方差
# v2 = np.var(a, axis=1, ddof=1) #偏样本方差
# print(np.mean(v1))
# print(np.mean(v2))
# =============================================================================

# =============================================================================
# #大小与排序
# a = np.array([1, 3, 5, 7]) 
# b = np.array([2, 4, 6])
# print(np.maximum(a[None, :], b[:, None]))
# #用argmax()和argmin()可以求最大值和最小值的下标
# np.random.seed(42)
# a = np.random.randint(0, 10, size=(4, 5)) 
# print(a)
# max_pos = np.argmax(a) 
# print(max_pos) #多个最值时得到第一个最值的下标
# #查看a平坦化之后的数组中下标为max_pos的元素
# print(a.ravel()[max_pos])
# print(np.max(a))
# #通过unravel_index()将一维数组的下标转换为多维数组的下标
# #第一个参数为一 维数组的下标，第二个参数是多维数组的形状
# idx = np.unravel_index(max_pos, a.shape)
# print(idx, a[idx])
# #在数组a中第0行的最大值的下标为2,
# #第1行的最大值的下标为0
# print(np.argmax(a, axis=1))
# #print(a[np.arange(a.shape[0]), idx]) #报错
# print('\n')
# #sort()函数则返冋一个新数组, 不改变原始数组
# print(np.sort(a)) #aixs默认为-1，沿着数组的最终轴进行排序(行排列)
# print(np.sort(a, axis=0)) #对数组a每列上的数值进行排序
# #argsort()返回数组的排序下标，参数axis的默认值为-1
# sort_axis1 = np.argsort(a)
# sort_axis0 = np.argsort(a, axis=0)
# print(sort_axis1)
# print(sort_axis0)
# print('\n')
# axis0, axis1 = np.ogrid[:a.shape[0], :a.shape[1]]
# print(a[axis0, sort_axis1])
# print(a[sort_axis0, axis1])
# print('\n')
# #exsort()返回排序下标，注意数组中最后的列为排序的主键。
# #names = ["zhang", "wang", "li", "wang", "zhang"] 
# #ages = [37, 33, 32, 31, 36] 
# #idx = np.lexsort([ages, names]) 
# #sorted_data = np.array(zip(names, ages), "0")[idx]
# #print(idx)
# #print(sorted_data)
# #b = np.random.randint(0, 10, (5, 3))
# #print(b)
# #print('\n')
# #print(b[np.lexsort(b[:, ::-1].T)])
# ##partition()和argpaitition()对数组进行分割,可以很快地找出排序之后的前k个元素
# ##由于它不需要对整个数组进行完整排序,速度比调用sort()之后再取前k个元素要快许多
# #r = np.random.randint(10, 1000000, 1000000)
# #print(np.sort(r)[:5])
# #print(np.partition(r, 5)[:5])
# #用％timeit测试sort()和partition()的运行速度:
# #%timeit np.sort(r)[:5]
# #%timeit np.sort(np.partition(r, 5)[:5])
# #75.7 ms ± 568 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# #8.35 ms ± 85.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# =============================================================================
# =============================================================================
# a = np.random.randint(0, 10, size=(4, 5)) 
# print(a)
# print(np.median(a, axis=1))
# print('\n')
# r = np.abs(np.random.randn(100000)) 
# print(np.percentile(r, [68.3, 95.4, 99.7]))
# #searchsorted()返回一个下标数组,将v中对应的元素插入到a中的位置,能够保持数据的升序排列
# a = [2, 4, 8, 16, 16, 32]
# v = [1, 5, 33, 6]
# print(np.searchsorted(a, v))
# print(np.searchsorted(a, v, side="right"))
# #searchsorted()可以用于在两个数组中查找相同的元素
# #找到y中每个元素在x中的下标,若不存在，将下标设置为-1
# x = np.array([3, 5, 7, 1, 9, 8, 6, 10])
# y = np.array([2, 1, 5, 10, 100, 6])
# 
# def get_index_searchsorted(x, y): 
#     index = np.argsort(x)  #获得升序排序的下标index
#     sorted_x = x[index] 
#     sorted_index = np.searchsorted(sorted_x, y)
#     "将下标限定在0到len(x)-1之间"
#     yindex = np.take(index, sorted_index, mode="clip") 
#     mask = x[yindex] != y  
#     yindex[mask] = -1
#     return yindex
# 
# print(get_index_searchsorted(x,y)) #返回y中与x相同数的下标，不存在返回-1
# print('\n')
# #统计函数
# #unique:去除重复元素, bincount对整数数组的元素计数
# #histogram:一维直方图统计, digitze:离散化
# #unique:去除重复元素
# np.random.seed(42)
# a = np.random.randint(0, 8, 10)
# print(a)
# print(np.unique(a))
# #retun_index: Ture表示同时返回原始数组中的下标
# x, index = np.unique(a, return_index=True)
# print(x)
# print(index)
# print(a[index])
# #retun_inverse: True表示返回重建原始数组用的下标数组
# x, rindex = np.unique(a, return_inverse=True)
# print(rindex)
# print(x[rindex])
# print('\n')
# #bincount:对整数数组的元素计数,返回数组中第i个元素的值表示整数i出现的次数
# np.bincount(a)
# x = np.array([0, 1, 2, 2, 1, 1, 0])
# w = np.array([0.1, 0.3, 0.2, 0.4, 0.5, 0.8, 1.2])
# print(np.bincount(x, w))
# a = np.random.rand(1000000)
# #bins指定统计的区间个数，即对统计范围的等分数
# print(np.histogram(a, bins=5, range=(0, 1))) #足够大时，近似相等
# #需要统计的区间的长度不等
# print(np.histogram(a, bins=[0, 0.4, 0.8,1.0]))
# #分段函数
# #where()函数可以看作判断表达式(x = y if condition else z)的数组版本
# x = np.arange(10)
# #print(x)
# #print(x.where(x < 5, 9-x, x))
# #piecewise()专门用于计算分段函数，
# #操作多维数组
# a = np.arange(3)
# b = np.arange(10, 13)
# v = np.vstack((a, b)) #连成一行
# h = np.hstack((a, b)) #连成一列
# c = np.column_stack((a, b)) #按列连接多个一维数组
# print(v)
# print(h)
# print(c)
# print(np.c_[a, b, a+b]) #按列连接数组
# #split()只能平均分组，而array_split()能尽量平均分组：
# np.random.seed(42)
# a = np.random.randint(0, 10, 12)
# print(np.split(a, 6))
# print(np.array_split(a, 5))
# a = np.random.randint(0, 10, (2, 3, 4, 5)) #生成6个4行五列的数组，数组按二行三列排
# print(a) 
# print("原数组形状：", a .shape)
# print("transpose:", np.transpose(a, (1, 2, 0, 3)).shape)
# print("swapaxes:", np.swapaxes(a, 1, 2).shape)
# =============================================================================

# =============================================================================
# #多项式函数
# a = np.array([1.0, 0, -2, 1])
# p = np.poly1d(a)
# print(type(p))
# print(p(np.linspace(0, 1, 5)))
# print(np.linspace(0, 1, 5))
# #对polyld对象进行:加减乘除运算相当于对相应的多项式函数进行计算
# print(p + [-2, 1]) #和 p + np.polyld([-2, 1])相同
# print(p * p)
# print(p / [1, 1]) #除法返回两个多项式，分别为商式和余式
# print(p == np.poly1d([1., -1, -1]) * [1,1] + 2)
# #多项式对象的deriv()和integ()方法分別计算多项式函数的微分和积分:
# print(p.deriv())
# print(p.integ())
# print(p.integ().deriv() == p)
# #多项式函数的根可以使用roots()函数计算：
# r = np.roots(p)
# print(r)
# print(p(r))
# #poly()函数可以将根转换回多项式的系数
# print(np.poly(r))
# print('\n')
# #pdyfit()函数可以对一组数据使用多项式函数进行拟合
# np.set_printoptions(suppress=True, precision=4)
# 
# x = np.linspace(-np.pi / 2, np.pi / 2, 1000) 
# y = np.sin(x)
# 
# for deg in [3, 5, 7]:
#     a = np.polyfit(x, y, deg)  
#     error = np.abs(np.polyval(a, x) - y) 
#     print("degree {}: {}".format(deg, a))
#     print("max error of order %d:" % deg, np.max(error))
# =============================================================================

# =============================================================================
# #多项式函数类
# #numpy.polynomial模块中提供了更丰富的多项式函数类
# #Polynomial、Chebyshev(切比雪夫)、 Legendre等。它们和前面介绍的numpy.polyld相反
# #多项式各项的系数按照幂从小到大的顺序
# #使用Polynomial类表示多项式x^3 - 2x + 1，并计算x = 2处的值:
# from numpy.polynomial import Polynomial, Chebyshev
# p = Polynomial([1, -2, 0, 1]) 
# print(p(2.0))
# print(p.deriv())
# print(Chebyshev.basis(4).convert(kind=Polynomial))
# def f(x):
#     return 1.0/ (1+25* x**2) 
# n = 11
# xl = np.linspace(-1, 1, n) 
# x2 = Chebyshev.basis(n).roots()  
# xd = np.linspace(-1, 1, 200)
# cl = Chebyshev.fit(xl, f(xl), n - 1, domain=[-1, 1])
# c2 = Chebyshev.fit(x2, f(x2), n - 1, domain=[-1, 1])
# print("插值多项式的最大误差：")
# print("等距离取样点：" + str(abs(cl(xd) - f(xd)).max()))
# print("切比雪夫节点：" + str(abs(c2(xd) - f(xd)).max()))
# 
# def g(x):
#     x = (x - 1) * 5
#     return(np.sin(x**2) + np.sin(x)**2) 
# n = 100
# x = Chebyshev.basis(n).roots() 
# xd = np.linspace(-1, 1, 1000)
# p_g = Polynomial.fit(x, g(x), n - 1, domain=[-1, 1]) 
# c_g = Chebyshev.fit(x, g(x), n - 1, domain=[-1, 1])
# print("Max Polynomial Error:" + str(abs(g(xd) - p_g(xd)).max()))
# print("Max Chebyshev Error:" + str(abs(g(xd) - c_g(xd)).max()))
# #trim()方法可以降低多项式的次数，将尾部绝对值小于参数tol的高次系数截断
# c_trimed = c_g.trim(tol=0.05)
# print("degree:" + str(c_trimed.degree()))
# print("error:" + str(abs(g(xd) - c_trimed(xd)).max()))
# def h(x):
#     x = 5 * x
#     return(np.exp(-x**2 / 10) )
# n = 20
# x = Chebyshev.basis(n).roots()
# c_h = Chebyshev.fit(x, h(x), n - 1, domain=[-1, 1])
# print("Max Chebyshev Error:" + str(abs(h(xd) - c_h(xd)).max()))
# 
# #多项式类支持四则运算
# c_diff = c_g - c_h
# roots = c_diff.roots()
# real_roots = roots[roots.imag == 0].real
# print(np.allclose(c_diff(real_roots), 0))
# 
# def g2(x):
#     return(np.sin(x**2) + np.sin(x)**2) 
# n = 100
# x = Chebyshev.basis(n, domain=[-10, 0]).roots()
# xd = np.linspace(-10, 0, 1000)
# c_g2 = Chebyshev.fit(x, g2(x), n - 1, domain=[-10, 0]) 
# print("Max Chebyshev Error:" + str(abs(g2(xd) - c_g2(xd)).max()))
# =============================================================================
# =============================================================================
# #各种乘积运算
# a = np.array([1, 2, 3])
# print(a[:, None])
# print(a[None, :])
# a = np.arange(12).reshape(2, 3, 2) 
# b = np.arange(12, 24).reshape(2, 2, 3) 
# c = np.dot(a, b) 
# print(a)
# print('\n')
# print(b)
# print(c)
# print(c.shape)
# 
# a = np.arange(12).reshape(2, 3, 2) 
# b = np.arange(12, 24).reshape(2, 3, 2) 
# c = np.inner(a, b) 
# print(c.shape)
# 
# a = np.array([1, 2, 3]) 
# b = np.array([4, 5, 6, 7])
# print(np.outer(a, b)) #计算列向量和行向量的矩阵乘积
# print(np.dot(a[:, None], b[None, :]))
# 
# a = np.random.rand(3, 4) 
# b = np.random.rand(4, 5)
# cl = np.tensordot(a, b, axes=[[1], [0]])
# c2 = np.tensordot(a, b, axes=1)
# c3 = np.dot(a, b) 
# assert np.allclose(cl, c3) #若为假返回异常
# assert np.allclose(c2, c3)
# a = np.arange(12).reshape(2, 3, 2) 
# b = np.arange(12, 24).reshape(2, 2, 3) 
# cl = np.tensordot(a, b, axes=[[-1], [-2]]) 
# c2 = np.dot(a, b) 
# assert np.alltrue(cl == c2) 
# 
# a = np.random.rand(4, 5, 6, 7) 
# b = np.random.rand(6, 5, 2, 3)
# c = np.tensordot(a, b, axes=[[1, 2], [1, 0]])
# for i, j, k, l in np.ndindex(4, 7, 2, 3):
#     assert np.allclose(c[i, j, k, l], np.sum(a[i, :, :, j] * b[:, :, k, l].T))
# print(c.shape) 
# =============================================================================

# =============================================================================
# #广义ufunc函数
# a = np.random.rand(10, 20, 3, 3)
# ainv = np.linalg.inv(a)
# #print(a)
# #print(ainv)
# print(ainv.shape)
# i, j = 3, 4
# print(np.allclose(np.dot(a[i, j], ainv[i, j]), np.eye(3)))
# adet = np.linalg.det(a) #计算矩阵的行列式
# print(adet.shape)
# n = 10000
# np.random.seed(0)
# beta = np.random.rand(n, 3)
# x = np.random.rand(n, 10)
# y = beta[:, 2, None] + x*beta[:, 1, None] + x**2*beta[:, 0, None]
# print(beta[42])
# print(np.polyfit(x[42], y[42], 2))
# #在numpy.polyfit()内部实际上是通过调用最小二乘法函数
# #numpy.linalg.lstsq()来实现多项式拟合的
# #我们也可以直接调用lstsq()计算系数：
# xx = np.column_stack(([x[42]**2, x[42], np.ones_like(x[42])]))
# print(np.linalg.lstsq(xx, y[42])[0])
# print('\n')
# X = np.dstack([x**2, x, np.ones_like(x)])
# Xt = X.swapaxes(-1, -2)
# beta2 = np.vstack([np.polyfit(x[i], y[i], 2) for i in range(n)])
# import numpy.core.umath_tests as umath
# A = umath.matrix_multiply(Xt, X)
# b = umath.matrix_multiply(Xt, y[..., None]).squeeze()
# beta3 = np.linalg.solve(A, b)
# print(np.allclose(beta3, beta2))
# M = np.array([[[np.cos(t), -np.sin(t)],
#                 [np.sin(t), np.cos(t)]]
#             for t in np.linspace(0, np.pi, 4, endpoint=False)])
# x = np.linspace(-1, 1, 100)
# points = np.array((np.c_[x, x], np.c_[x, x**3], np.c_[x**3, x]))
# rpoints = umath.matrix_multiply(points, M[:, None, ...]) 
# print(points.shape, M.shape, rpoints.shape)  
# =============================================================================
             
#实用技巧
#动态数组
#Python标准库中的array数组提供了动态分配内存的功能
#先用array数纽收集数据，然后通过np.frombuffer()
#将array数组的数据内存直接转换为NumPy数组
from array import array
a = array("d", [1,2,3,4]) #创建一个 array数组
#通过np.frombuffer()创建一个和a共享内存的NumPy数组
na = np.frombuffer(a, dtype=np.float) 
print(a)
print(na) 
na[1] = 20 #修改NumPy数组中下标为1的元素
print(a)
print('\n')
import math 
buf = array("d") 
for i in range(5):
    buf.append(math.sin(i*0.1)) 
    buf.append(math.cos(i*0.1))
data = np.frombuffer(buf, dtype=np.float).reshape(-1,2)
print(data)

#bytearray对象的+=运算与其extend()方法的功能相同
#但+=的运行速度要比extend〇快许多
import struct 
buf = bytearray() 
for i in range(5):
    buf += struct.pack("=hdd", i, math.sin(i*0.1), math.cos(i*0.1))
#通道i是短整型数，其类型符号为“h”，通道2和3为双精度浮点数，其类型符号为“d”
dtype = np.dtype({"names":["id", "sin","cos"], "formats":["h", "d", "d"]})  
data = np.frombuffer(buf, dtype=dtype) 
print(data)

#和其他对象共享内存
#如果对象没有提供该接口，但是能够 获取数据存•储区的地址，
#可以通过ctypes和numpy.ctypeslib模块中提供的函数，创建与对象共享内存的数组

#与结构数组共享内存
#从结构数组获取某个字段时，得到的是原数组的视图
#但是如果获取多个字段，将得到一 个全新的数组，不与原数组共享内存
persontype = np.dtype({
        'names':['name', 'age', 'weight', 'height'],
        'formats':['S30', 'i', 'f', 'f']}, align=True)
a = np.array([("Zhang", 32, 72.5, 167.0),
              ("Wang", 24, 65.2, 170.0)], dtype=persontype)
print(a["age"].base is a)
print(a[["age", "height"]].base is None)
#为了创建结构数组的多字段视图，可以使用下面的fields_View()函数
#它通过原数组的dtype 属性创建视图数组的dtype对象
#然后通过ndairayO创建视图数组
def fields_view(arr, fields):
    dtype2 = np.dtype({name:arr.dtype.fields[name] for name in fields}) 
    return(np.ndarray(arr.shape, dtype2, arr, 0, arr.strides))
    
v = fields_view(a, ["age", "weight"])
print(v.base is a)
v["age"] += 10
print(a)
#dtype对象的fields属性是一个以字段名为键、以字段类型和字节偏移_M为值的字典，
#使用它创建新的dtype对象时，可以保持字段的偏移量
print(a.dtype.fields)
print(a.dtype)
print(v.dtype)
























