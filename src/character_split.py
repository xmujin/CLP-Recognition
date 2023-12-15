# -*- coding: utf-8 -*-
# @Time    : 2023-12-14
# @Author  : 向波
# @Sid     : 12103990437
# @File    : character.py
# @Description : 用于字符分割，使用投影法，并将分割好的图片交给另一个程序进行处理
# 车牌识别
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_projection(projection, title):
    plt.plot(projection)
    plt.title(title)
    plt.show()

def GetProjection(img):
    # 应用自适应阈值二值化
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 计算水平投影，即将每一行的元素相加
    projection = [np.sum(binary, axis=1), np.sum(binary, axis=0)]
    return projection

def FindSplitPos(projection, threshold=10, charctorWidth=20):
    # 垂直投影的数组
    v = projection[1]
    starts = []
    ends = []
    threshold = 3
    # 双指针法
    l = 0
    r = 0
    while r < len(v):
        if v[r] != 0:
            # 移动右指针
            r += 1
        else:  # 如果碰到了0,遍历0的个数，确定是否满足分割条件
            temp = r
            count = 0  # 保存0的个数
            while v[temp] == 0:  # 结束时，temp已经指向了不为0的位置
                count += 1
                temp += 1
                if temp >= len(v):
                    break
            if count >= threshold:
                if r - l >= charctorWidth:
                    starts.append(l)
                    ends.append(r)
                l = r + count  # 重新定位左指针的位置
                r = l
            else:  # 如果0的个数少于预定值，表明结束位置还未确定，将右指针移动到不为0的位置
                r += count
    if r - l >= charctorWidth:
        starts.append(l)
        ends.append(r)
    return [starts, ends]
    # todo 水平分割的位置

def segment_characters(img, start_indices, end_indices):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    characters = []
    for start, end in zip(start_indices, end_indices):
        character = binary[:, start:end]
        characters.append(character)
    return characters






if __name__ == "__main__":
    image_path = "../pic/test/abc.jpg"

    # 读取图像
    img = cv2.imread(image_path, 0)

    # 获取水平和垂直投影
    projection =  GetProjection(img)

    segmented_characters = segment_characters(img, *FindSplitPos(projection))
    # 显示分割后的字符
    for i, character in enumerate(segmented_characters):
        cv2.imshow(f"Character {i + 1}", character)
    cv2.waitKey(0)
