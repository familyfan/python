# -*- coding: utf-8 -*-
#将matplotlib绘制的图表嵌入Notebook中
#matplotlib inline #使用inline模式在Notebook中绘制的图表会自动动关闭
#在Notebook的多个单元格内操作同一幅图表，需要运行下面的魔法命令：
#config InlineBackend.close_figures = False

#使用pyplot模块绘图
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  #雅黑字体
# =============================================================================
# x = np.linspace(0, 10, 1000)
# y = np.sin(x)
# z = np.cos(x**2)
# 
# plt.figure(figsize=(8,4)) #指定宽度和高度
# #注:用dpi参数指定Figurc对象的分辨率,默认为80，本例的宽度为8*80=640个像素.
# plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2) #linewidth可以缩写成lw
# plt.plot(x,z,"b--",label="$cos(x^2)$")
# 
# plt.xlabel("Time(s)")
# plt.ylabel("Volt")
# plt.title("PyPlot First Example")
# plt.ylim(-1.2,1.2) #显示Y轴的范围
# plt.legend() #显示图示
# 
# plt.show()
# plt.savefig("test.png", dpi=120)
# 
# import io
# buf = io.BytesIO() #创建一个用来保存图像内容的BytesIO对象
# plt.savefig(buf, fmt="png") #将图像以 png 格式保存到 buf中
# print(buf.getvalue()[:20]) #显示图像内容的前20个字节
# =============================================================================
# =============================================================================
# 
# #面向对象方式绘图
# #将而向对象的绘图库包装成只使用函数的API
# #pyplot模块的内部保荐了当前图表以 及当前子阁等信息
# #可以使用gcf()和gca()获得这两个对象
# #gcf是Get Current Figure的缩写,获得的是表示图表的Figure对象.
# #gca是Get Current Axes的缩写,获得的则是表示子图的Axes对象.
# fig = plt.gcf()
# axes = plt.gca()
# print(fig,axes)
# 
# #配置属性
# plt.figure(figsize=(4, 3))
# x = np.arange(0, 5, 0.1)
# line = plt.plot(x, 0.05*x*x)[0]  #plot返回一个列表
# line.set_alpha(0.5)  #调用Line2D对象的set_*()方法来设置属性值
# 
# lines = plt.plot(x, np.sin(x), x, np.cos(x))
# plt.setp(lines, color="r", linewidth=4.0)
# 
# print(line.get_linewidth())
# print(plt.getp(lines[0], "color"))  #返回color属性
# #通过getp()查看Figure对象的属性
# f = plt.gcf()
# print(plt.getp(f))
# #Figure对象的axes属性楚一个列表，它保存图表中的所有子图对象。
# #下面的程序查看当前图表的axes属性，它就是gca()所获得的当前子图对象：
# print(plt.getp(f, "axes"), plt.getp(f, "axes")[0] is plt.gca())
# #用plt.getp()可以继续获取AxesSubplot对象的属性
# #例如它的lines属性为子图中的Line2D对象列表
# alllines = plt.getp(plt.gca(), "lines")
# #其中的第一条曲线就是最开始
# print(alllines, alllines[0] is line)  
# #直接获取对象的属性
# print(f.axes, len(f.axes[0].lines))
# =============================================================================


# =============================================================================
# #绘制多子图
# for idx, color in enumerate("rgbyck"):
#     plt.subplot(321+idx, axisbg=color)
# 
# #某个子图占据整行或整列
# plt.subplot(221)   #第一行的左图
# plt.subplot(222)   #第一行的右图
# plt.subplot(212)   #第二整行
# 
# plt.figure(1)
# plt.figure(2)
# ax1 = plt.subplot(121) #在图表2中创建子图1
# ax2 = plt.subplot(122) #在图表2中创建子图2
# 
# x = np.linspace(0, 3, 100) 
# for i in range(5):
#     plt.figure(1) #选择图表 1 
#     plt.plot(x, np.exp(i*x/3)) 
#     plt.sca(ax1) #选择图表2的子图1
#     plt.plot(x,np.sin(i*x))
#     plt.sca(ax2)  #选择图标2的子图2
#     plt.plot(x, np.cos(i*x))
# 
# fig, axes = plt.subplots(2, 3)
# [a, b, c], [d, e, f] = axes
# print(axes.shape)
# print(b)
# 
# fig = plt.figure(figsize=(6, 6))
# ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
# ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
# ax3 = plt.subplot2grid((3, 3), (1, 0), rowspan=2)
# ax4 = plt.subplot2grid((3, 3), (2, 1), colspan=2)
# ax5 = plt.subplot2grid((3, 3), (1, 1))
# 
# #配置文件
# #获取用户配置路径
# from os import path
# print(path.abspath(matplotlib.get_configdir()))
# #获得目前使用的配置文件的路径:
# print(path.abspath(matplotlib.matplotlib_fname()))
# 
# #print(matplotlib.rc_params())  #配置文件的读入
# matplotlib.rcParams["lines.marker"] = "o"
# plt.plot([1, 2, 3, 2])
# matplotlib.rc("lines", marker="x", linewidth=2, color="red")
# plt.plot([1, 2, 3, 2])
# matplotlib.rcdefaults()  #恢复到matplotlib载入时从配置文件读入的默认配置
# plt.plot([1, 2, 3, 2])
# #重新从配置文件载入最新的配置
# matplotlib.rcParams.update(matplotlib.rc_params())
# #注意：通过pyplot模块也可以使用rcParams,rc和rcdefaults
# =============================================================================

# =============================================================================
# from matplotlib import style
# print(style.available)
# #调用use()函数即可切换样式，例如下面使用ggplot样式绘图
# style.use("ggplot")
# x = np.arange(0, np.pi, 0.0001)
# plt.plot(x, np.sin(x))
# 
# #在图表中显示中文
# from matplotlib.font_manager import fontManager
# print(fontManager.ttflist[:6])
# 
# #ttflist是matplotlib的系统字体列表，其屮每个元素都是表示字体的Font对象，
# #下面的程序显示了第一个字体文件的全路径和字体名
# print(fontManager.ttflist[0].name)
# print(fontManager.ttflist[0].fname)
# 
# import os
# from os import path
# from matplotlib.font_manager import fontManager
# fig = plt.figure(figsize=(8, 7))
# ax = fig.add_subplot(111)
# plt.subplots_adjust(0, 0, 1, 1, 0, 0)
# plt.xticks([])
# plt.yticks([])
# x, y = 0.05, 0.05
# fonts = [font.name for font in fontManager.ttflist if
# #             "利用0S模块中的stat()获収字体文件的大小"
#              path.exists(font.fname) and os.stat(font.fname).st_size>1e6]
# font = set(fonts)
# dy = (1.0 - y) / (len(fonts) // 4 + (len(fonts)%4 != 0)) 
# =============================================================================
# =============================================================================
# #调用子图对象的text()在其中添加文字，注意文字必须是Unicode字符串
# #通过一个描述字体的字典指定文字的字体：'fontname'键对应的值就是字体名
# for font in fonts:
#     t = ax.text(x, y + dy / 2, "中文字体",
#                 {'fontname':font, 'fontsize':14}, transform=ax.transAxes)
#     ax.text(x, y, font, {'fontsize':12}, transform=ax.transAxes) 
#     x += 0.25 
#     if x >= 1.0: 
#         y += dy 
#         x = 0.05 
# plt.show()
# 
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
# t = np.linspace(0, 10, 1000)
# y = np.sin(t)
# plt.close("all")
# plt.plot(t, y)
# plt.xlabel("时间", FontProperties=font)
# plt.ylabel("振幅", FontProperties=font) 
# plt.title ("正弦波", FontProperties=font) 
# plt.show()
# 
# #在复制完字体文件之后，需要运行下面的语句重新创建字体列表
# from matplotlib.font_manager import _rebuild
# _rebuild()
# 
# #可以直接修改配置字典，设置默认字体
# plt.rcParams["font.family"] = "SimHei" 
# plt.plot([1,2,3]) 
# plt.xlabel(0.5, 0.5, "中文字体")
# =============================================================================


# =============================================================================
# #Artist对象
# #直接创建Artist对象进行绘图的流程如下：
# #(1) 创建Figure对象
# #(2) 为Figure对象创建一个或多个Axes对象
# #(3) 调用Axes对象的方法来创建各种简单类型的Artist对象
# #fig = plt.figure()
# #ax = fig.add_axes([0.15, 0.1, 0.7, 0.3])  #[left,bottom,width, height]
# #line = ax.plot([1, 2, 3], [1, 2, 1])[0]  #返回的是只有一个元素的列表
# #print(line is ax.lines[0])
# ##通过set_xlabel()设置其X轴上的标题
# #ax.set_xlabel("time")
# ##Axes、XAxis和Text类都从Artist继承，也可以调用它们的get_*()以获得相应的属性值
# #print(ax.get_xaxis().get_label().get_text())
# #
# ##Artist的属性
# ##fig = plt.figure()
# ##fig.patch.set_color("g")  # 设置背景色为绿色
# ##fig.canvas.draw()
# #line = plt.plot([1, 2, 3, 2, 1], lw=4)[0]
# #line.set_alpha(0.5)  #变成半透明
# #line.set(alpha=0.5, zorder=2) #控制绘图顺序
# 
# ##Figure容器
# #fig = plt.figure()
# #ax1 = fig.add_subplot(211)
# #ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.3])
# #print(ax1 in fig.axes and ax2 in fig.axes)
# #
# #for ax in fig.axes:
# #    ax.grid(True)
# 
# #from matplotlib.lines import Line2D 
# #fig = plt.figure() 
# ##Figure对象的坐标系是以图表的左下角为坐标原点(0,0),右上角的坐标为(1，1)
# #line1 = Line2D(
# #        [0, 1], [0, 1], transform=fig.transFigure, figure=fig, color="r") 
# #line2 = Line2D(
# #        [0, 1], [1, 0], transform=fig.transFigure, figure=fig, color="g") 
# #fig.lines.extend([line1, line2])
# 
# =============================================================================
#Axes容器
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.patch.set_facecolor("green")
#
#x, y = np.random.rand(2, 100)
#line = ax.plot(x, y, "-", color="blue", linewidth=2)[0]
#print(line is ax.lines[0])

#fig, ax = plt.subplots()
#n, bins, rects = ax.hist(np.random.randn(1000), 50, facecolor="blue")
#print(rects[0] is ax.patches[0])

#fig, ax = plt.subplots()
#rect = plt.Rectangle((1, 1), width=5, height=12)
#ax.add_patch(rect) #将rect添加进ax
##print(rect.get_axes() is ax)

#print(ax.get_xlim())  # ax的X轴范围为0到1,无法显示完整的rect
##print(ax.dataLim._get_bounds())  # 数据的范围和 rect的大小一致
#ax.autoscale_view()  #自动调整坐标轴的范围
#print(ax.get_xlim()) #于是X轴可以完整显示rect

#fig, ax = plt.subplots()
#t = ax.scatter(np.random.rand(20), np.random.rand(20))
#print(t, t in ax.collections)

#Axis容器
#包括坐标轴上的刻度线、刻度文本、坐标网格以及坐标轴标题等内容
#刻度包括主刻度和副刻度，分别通过get_major_ticks()和get_minor_ticks()方法获得
#每个刻度线都是一个XTick或YTick对象，它包括实际的刻度线和刻度文本
#get_ticklabels()和get_ticklines()方法来直接获得刻度线和刻度文本
#fig, ax = plt.subplots() #先创建一个子图并获得其X轴对象axis
#axis = ax.xaxis
#print(axis.get_ticklocs())  #获得axis对象的刻度位置的列表
#print(axis.get_ticklabels())  #获得刻度标签的列表
#print([x.get_text() for x in axis.get_ticklabels()])  # 获得刻度的文本字符串
#print(axis.get_ticklines())
#print(axis.get_ticklines(minor=True))  #获得副刻度线列表
##获得刻度线或刻度标签之后，可以设置其各种属性，下面设置刻度线为绿色粗线，
##文本为红色并且旋转45°
#for label in axis.get_ticklabels(): 
#    label.set_color("red") 
#    label.set_rotation(45) 
#    label.set_fontsize(16)
#
#for line in axis.get_ticklines(): 
#    line.set_color("green") 
#    line.set_markersize(25) 
#    line.set_markeredgewidth(3)
#
#fig
#注意：xticks()只能设置刻度文本的属性，不能设置刻度线的属性
#print(axis.get_minor_locator()) #计算副刻度位置的对象
#print(axis.get_major_locator())  #计算主刻度位置的对象

##matplotlib提供了多种配置刻度线位置的Locator类和控制刻度文本显示的Formatter类
#from fractions import Fraction
#from matplotlib.ticker import MultipleLocator, FuncFormatter
#x = np.arange(0, 4*np.pi, 0.01)
#fig, ax = plt.subplots(figsize=(8,4))
#plt.plot(x, np.sin(x), x, np.cos(x))
#def pi_formatter(x, pos):
#    frac = Fraction(int(np.round(x / (np.pi/4))), 4) 
#    d, n = frac.denominator, frac.numerator 
#    if frac == 0:
#        return "0"
#    elif frac == 1: 
#        return "$\pi$" 
#    elif d == 1:
#        return(r"${%d} \pi$" % d)
#    elif n == 1:
#        return(r"$\frac{\pi}{%d}$" % d) 
#    return r"$\frac{%d \pi}{%d}$" % (n, d)
##设贾两个坐标轴的范围 
#plt.ylim(-1.5, 1.5) 
#plt.xlim(0, np.max(x))
##设置图的底边距
#plt.subplots_adjust(bottom = 0.15) 
#plt.grid() #开启网格
##主刻度为pi/4
#ax.xaxis.set_major_locator( MultipleLocator(np.pi/4) )
##主刻度文本用pi_formatter函数计算
#ax.xaxis.set_major_formatter(FuncFormatter( pi_formatter))
##副刻度为pi/20
#ax.xaxis.set_minor_locator( MultipleLocator(np.pi/20) )
##设置刻度文本的大小
#for tick in ax.xaxis.get_major_ticks(): 
#    tick.label1.set_fontsize(16)


##Artist对象的关系
#fig = plt.figure()
#plt.subplot(211)
#plt.bar([1, 2, 3], [1, 2, 3])
#plt.subplot(212)
#plt.plot([1, 2, 3])
##from scpy2.common import GraphvizMatplotlib
##%dot GraphvizMatplotlib.graphviz(fig)


##坐标变换和注释
#def func1(x):
#    return(0.6*x + 0.3)
#
#def func2(x): 
#    return(0.4*x*x + 0.1*x +0.2)
#    
#def find_curve_intersects(x, y1, y2):
#    d = y1 - y2
#    idx = np.where(d[:-1]*d[1:]<=0)[0] 
#    x1, x2 = x[idx], x[idx+1] 
#    d1, d2 = d[idx], d[idx+1] 
#    return(-d1*(x2-x1)/(d2-d1) + x1)
#
#x = np.linspace(-3, 3, 100) 
#f1 = func1(x) 
#f2 = func2(x)
#fig, ax = plt.subplots(figsize=(8,4)) 
#ax.plot(x, f1) 
#ax.plot(x, f2)
##计算两条曲线fl和f2的交点所对应的X轴坐标xl和x2
#x1, x2 = find_curve_intersects(x, f1, f2)
#ax.plot(x1, func1(x1), "o")
#ax.plot(x2, func2(x2), "o")
##绘制X轴上在两个交点之间、Y轴上在两条肋线之间的面积部分
##并通过facecolor和alpha参数指定填充的颜色和透明度
#ax.fill_between(x, f1, f2, where=f1>f2, facecolor="green", alpha=0.5)
#
#from matplotlib import transforms
##矩形区间使用fill_between()绘制
#trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#ax.fill_between([x1, x2], 0, 1, transform=trans, alpha=0.1)
#
#a = ax.text(0.05, 0.95, "直线和二次曲线的交点",
#            transform=ax.transAxes,
#            verticalalignment = "top",
#            fontsize = 18,
#            bbox={"facecolor":"red", "alpha":0.4, "pad":10}
#            )
#
#arrow = {"arrowstyle":"fancy, tail_width=0.6",
#         "facecolor":"gray",
#         "connectionstyle":"arc3, rad=-0.3"}
#
#ax.annotate("交点",
#            xy=(x1, func1(x1)), xycoords="data",
#            xytext=(0.05, 0.5), textcoords="axes fraction",
#            arrowprops = arrow)
#
#ax.annotate("交点",
#            xy=(x2, func1(x2)), xycoords="data",
#            xytext=(0.05, 0.5), textcoords="axes fraction",
#            arrowprops = arrow)
#
#xm = (x1+x2) / 2
#ym = (func1(xm) - func2(xm))/2 + func2(xm)
#o = ax.annotate("直线大于曲线区域",
#                xy=(xm, ym), xycoords="data",
#                xytext=(30, -30), textcoords="offset points",
#                bbox={"boxstyle":"round", "facecolor":(1.0, 0.7, 0.7), "edgecolor":"none"},
#                fontsize=16,
#                arrowprops={"arrowstyle":"->"}
#                )
#
#
##四种坐标系
##Axes对象的transData属性是数据坐标变换对象, transAxes属性是子图坐标变换对象
##Figure对象的transFigure属性是图表坐标变换对象
##通过上述坐标变换对象的transform()方法可以将此坐标系下的坐标转换为窗口坐标系中的坐标
#print(type(ax.transData))
#print(ax.transData.transform([(-3, -2), (3, 5)]))
##子图的左下角坐标(0,0)和数据坐标系中的坐标(-3, -2)在屏幕上是一个点
#print(ax.transAxes.transform([(0, 0), (1, 1)]))
##计算图表坐标系中坐标点(0,0)和(1,1)在绘图窗口中的位置
##可以看出绘图区域的宽为576个像素,高为288个像素：
#print(fig.transFigure.transform([(0,0), (1,1)]))
##通过坐标变换对象的inverted()方法，可以获得它的逆变换对象
##例如：计算绘图窗口中的坐标点(320,160)在数据坐标系中的坐标
##结果为(0.36666667 1.73287712):
#inv = ax.transData.inverted()
#print(type(inv))
#print(inv.transform((320, 160)))
##当调用set_xlim()修改子图所显示的X轴范围之后,它的数据坐标变换对象也同时发生了变化
#print(ax.set_xlim(-3, 2))  # 设置 X 轴的范围为-3 到 2
#print(ax.transData.transform((3, 5)))  #数据坐标变换对象已经发生了变化
#
##坐标变换的流水线
##在transAxes对象内部使用了transFigure变换
#print(ax.transAxes._boxout._transform == fig.transFigure)
##transLimits的最终效果就是将矩形区域(-3, -2)-(2, 5)变换为矩形区域(0, 0)-(1,1)
#print(ax.transLimits.transform((-3, -1.7799999999999998)))  #本电脑实际图片的坐标
#print(ax.transLimits.transform((2, 4.380000000000001)))
#print(ax.get_xlim())  #获得X轴的显示范围
#print(ax.get_ylim())  #获得Y轴的显示范圃
#
#t = ax.transLimits + ax.transAxes
#print(t.transform((0, 0)))
#print(ax.transData.transform((0, 0)))
#
##制作阴影效果
#fig, ax = plt.subplots() 
#x = np.arange(0., 2., 0.01) 
#y = np.sin(2*np.pi*x)
#N = 7 #阴影的条数
#for i in range(N, 0, -1):
#    offset = transforms.ScaledTranslation(i, -i, transforms.IdentityTransform())
#    "在完成数据坐标到窗口坐标的变换之后，再进行偏移变换"
#    shadow_trans = plt.gca().transData + offset
#    ax.plot(x, y, linewidth=4, color="black",
#            transform=shadow_trans,
#            alpha=(N-i)/2.0/N)
#ax.plot(x, y, linewidth=4, color="black")       
#ax.set_ylim((-1.5, 1.5))
#
#print(offset.transform((0, 0)))  # 将(0, 0)变换为(1,-1)
#print(ax.transData.transform((0,0)))  # 对(0,0)进行数据坐标变换
#print(shadow_trans.transform((0,0)))  #对(0,0)进行数据坐标变换和偏移变换
#
##添加注释
x = np.linspace(-1,1,10) 
y = x**2
fig, ax = plt.subplots(figsize=(8,4)) 
ax.plot(x,y)
for i, (_x, _y) in enumerate(zip(x, y)):
    ax.text(_x, _y, str(i), color="red", fontsize=i+10)
ax.text(0.5, 0.9, u"子图坐标系中的文字", color="blue", ha="center",
        transform=ax.transAxes)
plt.figtext(0.1, 0.92, u"图表坐标系中的文字", color="green")



#4.4块、路径和集合
#创建一个左下角位于(0,1)、宽为2、高为1的Rectangle矩形对象
#rect_patch = plt.Rectangle((0, 1), 2, 1)
#rect_path = rect_patch.get_path()
#print(rect_path.vertices)
#print(rect_path.codes)
#print('\n')
##通过get_patch_transfrom()获得Patch的坐标变换对象
##并用它将单位矩形的顶点坐标变换为我们所创建矩形的顶点坐标
#tran = rect_patch.get_patch_transform() 
#print(tran.transform(rect_path.vertices))

#from scpy2.matplotlib.svg_path import read_svg_path 
#ax = plt.gca()
#patches = read_svg_path("python-logo.svg")
#for patch in patches: 
#    ax.add_patch(patch)
#ax.set_aspect("equal") 
#ax.invert_yaxis() 
#ax.autoscale()

#集合




































































































































































































