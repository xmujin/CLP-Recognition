# -*- coding: utf-8 -*-
# @Time    : 2023-12-14
# @Author  : 向波
# @Sid     : 12103990437
# @File    : plate_right.py
# @Description : 用于将车牌进行矫正
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def TransVertexs(v: list):
    A, B, C, D = v
    width_AB = np.sqrt(((A[0] - B[0]) ** 2) + ((A[1] - B[1]) ** 2))
    width_CD = np.sqrt(((C[0] - D[0]) ** 2) + ((C[1] - D[1]) ** 2))
    maxWidth = max(int(width_AB), int(width_CD))
    height_AC = np.sqrt(((A[0] - C[0]) ** 2) + ((A[1] - C[1]) ** 2))
    height_BD = np.sqrt(((B[0] - D[0]) ** 2) + ((B[1] - D[1]) ** 2))
    maxHeight = max(int(height_AC), int(height_BD))
    output= np.float32([[0, 0],[maxWidth, 0],[0, maxHeight],[maxWidth, maxHeight]])
    return output, maxWidth, maxHeight

def GetRightPlate(image, baseName):
    """
    将车牌的图片弄正
    :param image: 传入的图片
    :param baseName: 传入的图片名
    :return:
    """
    # 根据图像名分割标注，第一个忽略，依次为角度，边框点，顶点，车牌号码，亮度，模糊度
    _, angle, boxPoints, vertexs, clpNubmer, brightness, blur = baseName.split('-')
    vertexs = vertexs.split("_")
    vertexs = [list(map(float, vertex.split('&'))) for vertex in vertexs]
    vertexsTrans = [vertexs[2], vertexs[3], vertexs[1], vertexs[0]]
    sb, w, h = TransVertexs(vertexsTrans)
    # 定义车牌的四个顶点坐标，顺序为左上、右上、右下、左下
    plate = np.array(vertexsTrans, dtype=np.float32)
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(plate, sb)
    result = cv2.warpPerspective(image, M, (w, h))  # 替换为你期望的输出尺寸
    return result

if __name__ == "__main__":
    filePath = "../dataset/plate_src/00891522988505-91_85-287&466_446&530-448&532_286&523_280&466_442&475-0_0_25_25_33_3_27-131-35.jpg"
    baseName = os.path.basename(filePath)
    img = cv2.imread(filePath)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cur = plate.copy()
    ############################################## 灰度图像的处理
    #cur = cv2.GaussianBlur(cur, (5, 5), 0)
    # 使用拉普拉斯滤波器增强边缘

    _, cur = cv2.threshold(cur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 定义开运算的核
    kernel = np.ones((3, 3), np.uint8)

    # 进行开运算
    cur = cv2.morphologyEx(cur, cv2.MORPH_OPEN, kernel)


    ############################################
    plates = GetRightPlate(cur, baseName)


    plt.subplot(1, 3, 1)
    plt.imshow(imgs)
    plt.axis("off")  # 不显示坐标轴
    plt.subplot(1, 3, 2)
    plt.imshow(plate, cmap='gray')
    plt.axis("off")  # 不显示坐标轴
    plt.subplot(1, 3, 3)
    plt.imshow(plates, cmap='gray')
    plt.axis("off")  # 不显示坐标轴




    plt.show()

