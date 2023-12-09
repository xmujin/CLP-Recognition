# -*- coding: utf-8 -*-
# @Time    : 2023-12-8
# @Author  : 向波
# @Sid     : 12103990437
# @File    : show_label.py
# @Description : 该程序用于实现对图片的标注信息进行分割
import os

from PIL import Image, ImageDraw

# 02-90_90-248&360_507&437-509&443_232&452_234&357_511&348-0_0_5_30_27_32_30-139-38
# 02 不知道啥意思
# 90_90 表示水平倾斜角、垂直倾斜角
# 248&360_507&437 表示车牌边界框的左上角和右下角坐标
# 509&443_232&452_234&357_511&348 表示车牌四个顶点的坐标，（边界框是将车牌完整框起来的框）
# 0_0_5_30_27_32_30，表示了车牌号码
# 139表示亮度
# 38表示模糊度，越小越模糊


provinceList = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

wordList = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]


def DrawVertex(img, vertexs):
    """
    该函数通过pillow库绘制车牌的四个顶点到图片上
    :param img: 传入的图片
    :param vertexs: 传入的顶点列表
    """
    draw = ImageDraw.Draw(img)
    radius = 3
    # 绘制顶点
    for pointX, pointY in vertexs:
        # 计算圆的边界框坐标
        left_top = (pointX - radius, pointY - radius)
        right_bottom = (pointX + radius, pointY + radius)
        # 绘制圆点
        draw.ellipse([left_top, right_bottom], fill="red")
def DrawBox(img, boxPoints):
    draw = ImageDraw.Draw(img)
    leftTop, rightBottom = boxPoints
    draw.rectangle([leftTop, rightBottom], outline="#00FF00", width=3)

if __name__ == "__main__":
    filePath = "../pic/test/02-90_90-248&360_507&437-509&443_232&452_234&357_511&348-0_0_5_30_27_32_30-139-38.jpg"
    # 提取文件名
    baseName = os.path.basename(filePath)
    # 移除文件扩展名
    baseName = os.path.splitext(baseName)[0]

    # 分割标注信息
    # 根据图像名分割标注，第一个忽略，依次为角度，边框点，顶点，车牌号码，亮度，模糊度
    _, angle, boxPoints, vertexs, clpNubmer, brightness, blur = baseName.split('-')
    boxPoints = boxPoints.split("_")
    boxPoints = [tuple(map(float, point.split('&'))) for point in boxPoints]

    vertexs = vertexs.split("_")
    vertexs = [tuple(map(float, vertex.split('&'))) for vertex in vertexs]

    # 显示车牌信息
    # --- 读取车牌号
    clpNubmer = clpNubmer.split('_')
    # 省份缩写
    province = provinceList[int(clpNubmer[0])]
    # 车牌信息
    words = [wordList[int(i)] for i in clpNubmer[1:]]
    # 车牌号
    clpNubmer = province + ''.join(words)
    print(clpNubmer)



    # 打开图像
    img = Image.open(filePath)
    DrawBox(img, boxPoints)
    DrawVertex(img, vertexs)
    img.show()