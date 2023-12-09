# -*- coding: utf-8 -*-
# @Time    : 日期，如2023-12-9
# @Author  : 作者 何奇航
# @Sid     : 学号 12103990429
# @File    : 当前文件名 rough_locate_plates.py
# @Description : 实现图像的预处理，车牌的粗定位



import cv2 as cv
import numpy as np
import os


def LoadImage(path):
    """
    本函数用于加载原图像
        :param path: 图片地址
    """
    src = cv.imread(path)
    return src
#灰度拉伸方法

def GrayStretch(image):
    """
    对原图像进行灰度拉伸处理，扩展图像的直方图，可以有效减小亮度过高或者亮度过低对图像造成的影响。
            :param image: 灰度图像
    """

    #灰度图中最大的像素值
    max_value=float(image.max())
    #灰度图中最小的像素值
    min_value=float(image.min())
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            #拉抻处理
            image[i,j]=(255/(max_value-min_value)*image[i,j]-(255*min_value)/(max_value-min_value))
    return image

def ImageBinary(image):
    """
    对拉伸后的灰度图进一步二值化处理
        :param image: 灰度拉伸后的灰度图像
    """
    max_value=float(image.max())
    min_value=float(image.min())

    #阈值
    ret=max_value-(max_value-min_value)/2
    #二值化
    ret,thresh=cv.threshold(image,ret,255,cv.THRESH_BINARY)
    return thresh

def FindRectangle(contour):
    """
        矩形轮廓角点，寻找到矩形之后记录角点，用来做参考以及画图。
            :param contour: 矩形轮廓角点
    """
    y,x=[],[]
    for value in contour:
        y.append(value[0][0])
        x.append(value[0][1])
    return [min(y),min(x),max(y),max(x)]


def LocatePlate(image, after):
    """
        车牌定位方法，需要两个参数，第一个是用来寻找位置，第二个为原图，用来绘制矩形。寻找位置的图片为经过几次形态学操作的图片。这里利用权值的操作，实
现了定位的最高概率。
            :param image: 原始传入图像
    """
    #寻找轮廓
    contours,hierarchy=cv.findContours(image,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    img_copy = after.copy()
    #找出最大的三个区域
    solving=[]
    for c in contours:
        r=FindRectangle(c)

        #矩形面积
        a=(r[2]-r[0])*(r[3]-r[1])

        #矩形长宽之比
        s=(r[2]-r[0])/(r[3]-r[1])

        solving.append([r,a,s])
    #通过参考选出面积最大的区域
    solving=sorted(solving,key=lambda b: b[1])[-3:]
    #颜色识别
    maxweight,maxindex=0,-1
    for i in range(len(solving)):#
        wait_solve=after[solving[i][0][1]:solving[i][0][3],solving[i][0][0]:solving[i][0][2]]
        #BGR转HSV
        hsv=cv.cvtColor(wait_solve,cv.COLOR_BGR2HSV)
        #蓝色车牌的范围 Hsv色彩空间的设置。
        lower=np.array([100,50,50])
        upper=np.array([140,255,255])
        #利用inrange找出掩膜
        mask=cv.inRange(hsv,lower,upper)
        #计算权值用来判断。
        w1=0
        for m in mask:
            w1+=m/255
        w2=0
        for n in w1:
            w2+=n
        #选出最大权值的区域
        if w2>maxweight:
            maxindex=i
            maxweight=w2
    return solving[maxindex][0]
'''
框出车牌 获取位置坐标，并返回图像
'''
#对图像的预处理
def FindPlates(image):
    """
        框出车牌 获取位置坐标，并返回图像
            :param image: 原始传入RGB图像
    """
    image=cv.resize(image,(400,int(400 * image.shape[0] / image.shape[1])))
    #转换为灰度图像
    gray_image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    #灰度拉伸
    #如果一幅图像的灰度集中在较暗的区域而导致图像偏暗，可以用灰度拉伸功能来拉伸(斜率>1)物体灰度区间以改善图像；
    # 同样如果图像灰度集中在较亮的区域而导致图像偏亮，也可以用灰度拉伸功能来压缩(斜率<1)物体灰度区间以改善图像质量
    stretchedimage=GrayStretch(gray_image)#进行灰度拉伸，是因为可以改善图像的质量

    '''进行开运算，用来去除噪声'''
    #构造卷积核

    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(30,30))
    #开运算
    openingimage=cv.morphologyEx(stretchedimage,cv.MORPH_OPEN,kernel)
    #获取差分图，两幅图像做差  cv2.absdiff('图像1','图像2')
    strtimage=cv.absdiff(stretchedimage,openingimage)

    #图像二值化
    binaryimage=ImageBinary(strtimage)
    #canny边缘检测
    canny=cv.Canny(binaryimage,binaryimage.shape[0],binaryimage.shape[1])
    #5 24效果最好
    kernel=np.ones((5,24),np.uint8)
    closingimage=cv.morphologyEx(canny,cv.MORPH_CLOSE,kernel)
    openingimage=cv.morphologyEx(closingimage,cv.MORPH_OPEN,kernel)
    #11 6的效果最好
    kernel=np.ones((11,6),np.uint8)
    openingimage=cv.morphologyEx(openingimage,cv.MORPH_OPEN,kernel)
    #消除小区域，定位车牌位置
    rect=LocatePlate(openingimage, image)#rect包括轮廓的左上点和右下点，长宽比以及面积
    #展示图像
    cv.imshow('image',image)
    cv.rectangle(image, (rect[0]-5, rect[1]-5), (rect[2]+5,rect[3]+5), (0, 255, 0), 2)
    cv.imshow('after', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

#文件中大量图片依次处理
# def runing():
#     file_path='../picture/1.jpg'
#     for filewalks in os.walk(file_path):
#         for files in filewalks[2]:
#             print('正在处理',os.path.join(filewalks[0],files))
#             find_plates(load_image(os.path.join(filewalks[0],files)))


filePath = '../picture/2.jpg'
image = LoadImage(filePath)
result = FindPlates(image)
cv.imshow('1', result)
cv.waitKey(0)