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





# threshold 表示空有多少才能作为分割点, 字符宽度用于判断
# def FindVSplitPos(v, threshold=10, charctorWidth=20):
#     starts = []
#     ends = []
#     # 双指针法
#     l = 0
#     r = 0
#     while r < len(v):
#         if v[r] != 0:
#             # 移动右指针
#             r += 1
#         else:  # 如果碰到了0,遍历0的个数，确定是否满足分割条件
#             temp = r
#             count = 0  # 保存0的个数
#             while v[temp] == 0:  # 结束时，temp已经指向了不为0的位置
#                 count += 1
#                 temp += 1
#                 if temp >= len(v):
#                     break
#             if count >= threshold:
#                 if r - l >= charctorWidth:
#                     starts.append(l)
#                     ends.append(r)
#                 l = r + count  # 重新定位左指针的位置
#                 r = l
#             else:  # 如果0的个数少于预定值，表明结束位置还未确定，将右指针移动到不为0的位置
#                 r += count
#     if r - l >= charctorWidth:
#         starts.append(l)
#         ends.append(r)
#     return [starts, ends]

def FindVSplitPos(v, threshold=500, width=1, charctorWidth=10):
    starts = []
    ends = []
    # 双指针法
    l = 0
    r = 0
    while r < len(v):
        if v[r] >= threshold:
            # 移动右指针
            r += 1
        else:  # 如果小于阈值，确定是否满足分割条件
            temp = r
            count = 0  # 保存0的个数
            while v[temp] < threshold:  # 结束时，temp已经指向了不为0的位置
                count += 1
                temp += 1
                if temp >= len(v):
                    break
            if count >= width:
                if r - l >= charctorWidth:
                    starts.append(l)
                    ends.append(r)
                l = r + count  # 重新定位左指针的位置
                r = l
            else:  # 如果谷值的个数少于预定值，表明结束位置还未确定，将右指针移动到不为0的位置
                r += count
    if r - l >= charctorWidth:
        starts.append(l)
        ends.append(r)
    return [starts, ends]



# 查找水平分割的位置, threshold表示小于该值的都可称之为谷底, width 表示当连续为谷时候的判断
def FindHSplitPos(h, threshold=10000, width=30):
    start = 0
    end = 0
    # 双指针法
    l = 0
    r = 0
    while r < len(h):
        if h[r] <= threshold:
            # 移动右指针
            r += 1
        else:  # 如果碰到了大于threshold
            temp = r
            count = 0  # 保存大于于阈值的个数的个数
            while h[temp] > threshold:
                count += 1
                temp += 1
                if temp >= len(h):
                    break
            if count >= width:
                l = r
                r += count
                start = l
                end = r
            else:
                r += count
                l = r  # 移动左指针
    return [start, end]

# 传入二值图并分割
def SplitCharacters(binary):
    # 计算水平投影
    hProjection = np.sum(binary, axis=1)
    # 通过水平投影分割图片
    h_split = binary.copy()
    s, e = FindHSplitPos(hProjection)
    h_split = h_split[s : e, :]
    characters = []
    # 进行垂直分割
    # 计算投影,用分割好的计算
    vProjection = np.sum(h_split, axis=0)
    starts, ends = FindVSplitPos(vProjection)
    #print([starts, ends])
    # 分割并腐蚀
    kernel = np.ones((3, 3), np.uint8)
    for start, end in zip(starts, ends):
        c = h_split[:, start:end]
        #c = cv2.erode(c, kernel, iterations=1)
        characters.append(c)
    return characters


if __name__ == "__main__":
    image_path = "../dataset/last_dataset/sb/02-99_71-203&483_427&583-429&608_210&562_193&472_412&518-0_0_11_24_11_32_33-103-15.jpg"

    # 读取图像
    img = cv2.imread(image_path, 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    projection = [np.sum(img, axis=1), np.sum(img, axis=0)]
    # 腐蚀一下
    # kernel = np.ones((3, 3), np.uint8)
    # hh = cv2.erode(img, kernel, iterations=1)
    #img = cv2.erode(img, kernel, iterations=1)
    characters = SplitCharacters(img)

    plt.figure(1)
    # 获取水平和垂直投影
    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap='gray')
    # print(img)
    plt.subplot(2, 1, 2)
    plt.plot(projection[1])

    plt.figure(2)
    # 显示分割后的字符
    for i, character in enumerate(characters):
        plt.subplot(3, 3, i + 1)
        plt.imshow(character, cmap='gray')
    plt.show()
