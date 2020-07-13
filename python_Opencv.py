# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
#import cv2
from matplotlib.cm import ScalarMappable
from IPython.display import Image

#img = cv2.imread('fig_1.jpg')
#print(type, img.shape, img.dtype)  # <class 'type'> (1010, 1615, 3) uint8
#cv2.namedWindow("demo1")
#cv2.imshow("demo1", img)
##cv2.waitKey(0)
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print(img_gray.shape) # (1010, 1615)

"AttributeError: module 'numpy' has no attribute 'orgid'"
#img = cv2.imread("fig_1.jpg")
#for quality in [90, 60, 30]:
#    cv2.imwrite("fig_q{:02d}.jpg".format(quality), 
#                img, [cv2.IMWRITE_JPEG_QUALITY, quality])
#
#def func(x, y, a):
#    return (x*x - y*y) * np.sin((x+y)/a) / (x*x + y*y)
#
#def make_image(x, y, a, dtype="uint8"):
#    z = func(x, y, a)
#    img_rgba = ScalarMappable(cmap="jet").to_rgba(z)
#    img = (img_rgba[:, :, 2::-1] * np.iinfo(dtype).max).astype(dtype)
#    return img
#
#y, x = np.orgid[-10:10:250j, -10:10:500j]
#img_8bit = make_image(x, y, 0.5, dtype = "uint8")
#img_16bit = make_image(x, y, 0.5, dtype = "uint16")
#cv2.imwrite("img_8bit.jpg", img_8bit)
#cv2.imwrite("img_16bit.jpg", img_16bit)
#cv2.imwrite("img_8bit.png", img_8bit)    
#cv2.imwrite("img_16bit.png", img_16bit)   

#with open("fig_1.jpg", "rb") as f:
#    jpg_str = f.read()
#
#jpg_data = np.frombuffer(jpg_str, np.uint8)
#img = cv2.imdecode(jpg_data, cv2.IMREAD_UNCHANGED)
#res, png_data = cv2.imencode(".png", img)
#png_str = png_data.tobytes()
#Image(data=png_data.tobytes())



 
#import cv2 as cv
#import numpy as np
#from scipy import ndimage


#cv.resize(InputArray src, OutputArray dst, Size, fx, fy, interpolation)
#InputArray src:输入图片,
#OutputArray dst:输出图片, 
#Size: 输出图片尺寸(宽，高), 
#fx, fy: 沿x轴，y轴的缩放系数,
#interpolation：插入方式
#INTER_NEAREST：最近邻插值
#INTER_LINEAR：双线性插值（默认设置）
#INTER_AREA：使用像素区域关系进行重采样。
#INTER_CUBIC：4x4像素邻域的双三次插值
#INTER_LANCZOS4：8x8像素邻域的Lanczos插值


 
## 读入原图片
#img = cv.imread('fig_1.jpg')
# 打印出图片尺寸
#print(img.shape)  #(1010, 1615, 3)
## 将图片高和宽分别赋值给x，y
#x, y = img.shape[0:2]
# 
## 显示原图
#cv.imshow('OriginalPicture', img)
# 
## 缩放到原来的二分之一，输出尺寸格式为（宽，高）
#img_test1 = cv.resize(img, (int(y / 2), int(x / 2)))
#cv.imshow('resize0', img_test1)
#cv.waitKey()
# 
## 最近邻插值法缩放
## 缩放到原来的四分之一
#img_test2 = cv.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)
#cv.imshow('resize1', img_test2)
#cv.waitKey()
#cv.destroyAllWindows()

 
#img = np.zeros((300, 300))
##img[0, 0] = 255
#img[:, 10] = 255
#img[10, :] = 255
#cv.imshow("img_1", img)
#cv.waitKey(0)

# 卷积核与图像特征提取
#kernel33 = np.array([[-1,-1,-1],
#                  [-1,8,-1],
#                  [-1,-1,-1]])
#
#kernel33_D = np.array([[1,1,1],
#                  [1,-8,1],
#                  [1,1,1]])
#
#img = cv.imread("dog.jpg",0)
#linghtImg = ndimage.convolve(img,kernel33_D)
#cv.imshow("img",linghtImg)
#cv.waitKey() 

#卷积核
#def convolve(dateMat,kernel):
#    m,n = dateMat.shape
#    km,kn = kernel.shape
#    newMat = np.ones(((m - km + 1),(n - kn + 1)))
#    tempMat = np.ones(((km),(kn)))
#    for row in range(m - km + 1):
#        for col in range(n - kn + 1):
#            for m_k in range(km):
#                for n_k in range(kn):
#                    tempMat[m_k,n_k] = dateMat[(row + m_k),(col + n_k)] * kernel[m_k,n_k]
#            newMat[row,col] = np.sum(tempMat)
#
#    return newMat 
 


#利用Gauss模糊处理
#img = cv2.imread("dog.jpg",0)
#blurred = cv2.GaussianBlur(img,(11,11),0)
#gaussImg = img - blurred
#cv2.imshow("img",gaussImg)
#cv2.waitKey()




# 加载与现实图像
#src = cv.imread("dog.jpg")
##cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
##cv.imshow("input", src)
#
### 转换为灰度
#gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#cv.imshow("gray", gray)
#print(src.shape)
#print(gray.shape)
#cv.imwrite("gray.png", gray)
#
## 创建空白图像
#black = np.zeros_like(src)
#cv.imshow("black", black)
#cv.imwrite("black.png", black)
#
## 调节亮度
#black[:,:,:] = 50
#lighter = cv.add(src, black)
#darker = cv.subtract(src, black)
#cv.imshow("lightness", lighter)
#cv.imshow("darkness", darker)
#cv.imwrite("lightness.png", lighter)
#cv.imwrite("darkness.png", darker)
#
## 调节对比度
#dst = cv.addWeighted(src, 1.2, black, 0.0, 0)
#cv.imshow("contrast", dst)
#cv.imwrite("contrast.png", dst)
#
## scale
#h, w, c = src.shape
#dst = cv.resize(src, (h//2, w//2))
#cv.imshow("resize-image", dst)
#
## 左右翻转
#dst = cv.flip(src, 1)
#cv.imshow("flip", dst)
#
## 上下翻转
#dst = cv.flip(src, 0)
#cv.imshow("flip0", dst)
#cv.imwrite("flip0.png", dst)
#
## rotate
#M = cv.getRotationMatrix2D((w//2, h//2),45, 1)
#dst = cv.warpAffine(src, M, (w, h))
#cv.imshow("rotate", dst)
#cv.imwrite("rotate.png", dst)
#
## 色彩
## HSV
#hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
#cv.imshow("hsv", hsv)
#
## 色彩表 - 支持14种色彩变换
#dst = cv.applyColorMap(src, cv.COLORMAP_AUTUMN)
#cv.imshow("color table", dst)
#cv.imwrite("color_table.png", dst)
#
## blur
#blur = cv.blur(src, (15, 15))
#cv.imshow("blur", blur)
#cv.imwrite("blur.png", blur)
#
## gaussian blur
#gblur = cv.GaussianBlur(src, (0, 0), 15)
#cv.imshow("gaussian blur", gblur)
#cv.imwrite("gaussian.png", gblur)
#
## custom filter - blur
#k = np.ones(shape=[5, 5], dtype=np.float32) / 25
#dst = cv.filter2D(src, -1, k)
#cv.imshow("custom blur", dst)
#cv.imwrite("custom_blur.png", dst)
#
## EPF
#dst = cv.bilateralFilter(src, 0, 100, 10)
#cv.imshow("bi-filter", dst)
#cv.imwrite("bi_blur.png", dst)
#
## gradient
#dx = cv.Sobel(src, cv.CV_32F, 1, 0)
#dy = cv.Sobel(src, cv.CV_32F, 0, 1)
#dx = cv.convertScaleAbs(dx)
#dy = cv.convertScaleAbs(dy)
#cv.imshow("grad-x", dx)
#cv.imshow("grad-y", dy)
#cv.imwrite("grad.png", dx)
#
## edge detect
#edge = cv.Canny(src, 100, 300)
#cv.imshow("edge", edge)
#cv.imwrite("edge.png", edge)
#
## 直方图均衡化
#eh = cv.equalizeHist(gray)
#cv.imshow("eh", eh)
#cv.imwrite("eh.png", eh)
#
## 角点检测
#corners = cv.goodFeaturesToTrack(gray, 100, 0.05, 10)
## print(len(corners))
#for pt in corners:
#    # print(pt)
#    b = np.random.random_integers(0, 256)
#    g = np.random.random_integers(0, 256)
#    r = np.random.random_integers(0, 256)
#    x = np.int32(pt[0][0])
#    y = np.int32(pt[0][1])
#    cv.circle(src, (x, y), 5, (int(b), int(g), int(r)), 2)
#cv.imshow("corners detection", src)
#cv.imwrite("corners.png", src)
#
## 二值化
#src = cv.imread("zsxq/zsxq_12.jpg")
#gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#cv.imshow("binary input", gray)
#
## 固定阈值
#cv.threshold(src, dst, thresh, maxval, type)
#if src(x, y) > thresh = maxval , else = 0

#ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
#cv.imshow("binary", binary)
#cv.imwrite("binary.png", binary)
#
## 全局阈值
#ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#cv.imshow("otsu", binary)
#
## 自适应阈值
#binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 10)
#cv.imshow("ada", binary)
#cv.imwrite("ada.png", binary)
#
## 轮廓分析
#contours, hireachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#result = np.zeros_like(src)
#for cnt in range(len(contours)):
#    cv.drawContours(result, contours, cnt, (0, 0, 255), 2, 8)
#cv.imshow("contour", result)
#cv.imwrite("contour.png", result)
#
## 膨胀与腐蚀操作
#se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))
#d = cv.dilate(binary, se)
#e = cv.erode(binary, se)
#cv.imshow("dilate", d)
#cv.imshow("erode", e)
#
## 开闭操作
#op = cv.morphologyEx(binary, cv.MORPH_OPEN, se)
#cl = cv.morphologyEx(binary, cv.MORPH_CLOSE, se)
#cv.imshow("open", op)
#cv.imshow("close", cl)
#
#
#cv.waitKey(0)
#cv.destroyAllWindows()


#image = "200_1_0.jpg"
#
#img = cv.imread(image)
#w, h, c = img.shape
#print(img.dtype)
#channel = 0
#channel += np.sum(img[:, :, :])
##print(img[150:, 100:150, :])
#mean = channel / (w*h*c)
#print(mean.dtype)
#new_image = img - mean
#print(new_image[150:, 100:150, :])
#rnew_image = new_image.astype(np.uint8)
#print("rnew_image:", rnew_image[150:, 100:150, :])


"""
在图片上显示文本
"""
##include<cv.h>
##include <highgui.h> 
##include <iostream>  
#using namespace std;
#int main(int argc, char** argv)
#{
# IplImage *src1;
# src1 = cvLoadImage(argv[2],1);
# cvNamedWindow("1", CV_WINDOW_AUTOSIZE);
# CvFont  font;
# cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX,1.0, 1.0);
# cvPutText(src1, "Hello", cvPoint(200,200), &font, cvScalar(255, 0, 0));
#        cvShowImage("1", src1);
#        
# cvWaitKey(0);
# cvReleaseImage(&src1);
# return 0;
#}



"""
抓取轮廓并轮廓绘制
"""
#cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)



img = cv.imread('test.jpg')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.imshow('imageshow', imgray)  # 显示返回值image，其实与输入参数的thresh原图没啥区别
cv.waitKey()
img = cv.drawContours(img, contours, -1, (0, 255, 0), 5)  # img为三通道才能显示轮廓
cv.imshow('drawimg', img)
cv.imwrite("test_contours.jpg", img)
cv.waitKey()
cv.destroyAllWindows()


#参考链接:
#https://blog.csdn.net/qdbszsj/article/details/64907121?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-17.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-17.nonecase
#截取轮廓中的图像参考链接:
#https://blog.csdn.net/qq_40909394/article/details/83996379?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.nonecase

"""
利用mask分离出目标图像，去掉背景
"""
# =============================================================================
# import os
# import cv2
# import numpy as np
# 
# def add_mask2image_binary(images_path, masks_path, masked_path):
# # Add binary masks to images
#     for img_item in os.listdir(images_path):
#         # print(img_item)
#         img_path = os.path.join(images_path, img_item)
#         img = cv2.imread(img_path)
#         # print(img.shape) # (1080, 1080, 3)
#         mask_path = os.path.join(masks_path, img_item[:-4]+'.png')  # mask是.png格式的，image是.jpg格式的
#         # mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
#         # print(mask_data.shape) # (1080, 1080)
#         # print(mask_data[400:500, 400:500]) # 114
#         mask_data = cv2.imread(mask_path)
#         mask_data = cv2.cvtColor(mask_data, cv2.COLOR_BGR2GRAY)
#         # cv2.imwrite(os.path.join(masked_path, img_item[:-4] + '_binaryzation.png'), mask_data)
#         cv2.imwrite(os.path.join(masked_path, img_item[:-4] + '_binaryzation_1.png'), mask_data)
#         masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask_data)  #将image的相素值和mask像素值相加得到结果
#         # cv2.imwrite(os.path.join(masked_path, img_item), masked)
#         cv2.imwrite(os.path.join(masked_path, img_item[:-4] + '_1.jpg'), masked)
# 
# images_path = "/home/stu1/fanliming/cervial_cancer/datasets/test_images/cut_cat_mask/image/"
# masks_path = "/home/stu1/fanliming/cervial_cancer/datasets/test_images/cut_cat_mask/image_mask/"
# masked_path = "/home/stu1/fanliming/cervial_cancer/datasets/test_images/cut_cat_mask/masked/"
# 
# add_mask2image_binary(images_path, masks_path, masked_path)
# 
# =============================================================================



"""
利用json文件数据画出图片的轮廓
"""
# =============================================================================
# import cv2
# import numpy as np
# import json
# import os
# images_path = "/home/stu1/fanliming/cervial_cancer/datasets/test_images/cat_mouse/cat_mouse.png"
# json_path = "/home/stu1/fanliming/cervial_cancer/datasets/test_images/cat_mouse/cat_mouse.json"
# mask_images_path = "/home/stu1/fanliming/cervial_cancer/datasets/test_images/cat_mouse/"
# category_types = ["Background", "mouse", "cat"]
# 
# img = cv2.imread(images_path)
# h, w = img.shape[:2]
# mask = np.zeros([h, w, 1], np.uint8)    # 创建一个大小和原图相同的空白图像
# 
# with open(json_path, "r") as f:
#     label = json.load(f)
# 
# shapes = label["shapes"]
# 
# # for shape in shapes:
# #     category = shape["label"]
# #     points = shape["points"]
# #     # 填充
# #     points_array = np.array(points, dtype=np.int32)
# #     mask = cv2.fillPoly(mask, [points_array], category_types.index(category))
# #
# # cv2.imwrite(os.path.join(mask_images_path, "mask_image.png"), mask)
# 
# 
# for shape in shapes:
#     category = shape["label"]
#     points = shape["points"]
#     points_array = np.array(points, dtype=np.int32)
#     if category == "cat":
#         # 调试时将Tom的填充颜色改为200，便于查看
#         mask = cv2.fillPoly(mask, [points_array], 200)
#     else:
#         mask = cv2.fillPoly(mask, [points_array], 100)
# cv2.imwrite(os.path.join(mask_images_path, "mask_image_viz.png"), mask)
# =============================================================================

"cv2.fillPoly()"













