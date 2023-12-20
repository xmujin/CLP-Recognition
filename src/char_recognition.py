# -*- coding: utf-8 -*-
# @Time    : 2023-12-18
# @Author  : 向波
# @Sid     : 12103990437
# @File    : char_recognition.py
# @Description :使用字符模型，对分割好的字符进行识别
import cv2
import numpy as np
import torch
import torchvision

from torchvision.transforms import functional
from src.build_char_recognition_model import CharCNN

dataset = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵', '桂', '黑', '沪',
           '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼',
           '陕', '苏', '晋', '皖', '湘', '新', '豫', '渝', '粤', '云',
           '藏', '浙']


def GetCharModel(modelPath):
    """

    :param modelPath: 模型的路径
    :return: 评估的模型
    """
    num_classes = 67  # 背景和车牌两个类别
    model = CharCNN(num_classes)
    # 加载模型参数
    model.load_state_dict(torch.load(modelPath))
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    return model


# 读取图片
# img = cv2.imread("../images/cnn_char_train/0/4-3.jpg", cv2.IMREAD_GRAYSCALE)
# img = functional.to_tensor(img).unsqueeze(0).to(device)

# def GetChar(model, imgSrc):
#     """
#     识别字符
#     :param model: 使用的模型
#     :param imgSrc: 图片
#     :return: 识别的字符
#     """
#     print("start")
#     device = torch.device("cuda")
#     print("fnish1")
#     # 将原图像改为训练时用到的尺寸
#     img = cv2.resize(imgSrc, (20, 20))
#     print("fnish2")
#     img = functional.to_tensor(img).unsqueeze(0).to(device)
#     print("fnish3")
#     with torch.no_grad():
#         print("fnish3--")
#         prediction = model(img)
#     print("fnish4")
#     preClass = torch.argmax(prediction)
#     print("end")
#     return dataset[preClass.item()]
#


class CharRecModel:
    def __init__(self, modelPath):
        self.model = None
        self.load_model(modelPath)

    def load_model(self, modelPath):
        num_classes = 67  # 背景和车牌两个类别
        self.model = CharCNN(num_classes)
        # 加载模型参数
        self.model.load_state_dict(torch.load(modelPath))
        device = torch.device('cuda')
        self.model.to(device)
        self.model.eval()
        # 进行第一次预先推理，以成功加载模型
        arr = np.zeros((20, 20), dtype=np.float32)
        self.GetChar(arr)
    def GetChar(self, imgSrc):
        device = torch.device("cuda")
        # 将原图像改为训练时用到的尺寸
        img = cv2.resize(imgSrc, (20, 20))
        img = functional.to_tensor(img).unsqueeze(0).to(device)
        # 进行推理
        with torch.no_grad():
            prediction = self.model(img)
        preClass = torch.argmax(prediction)
        return dataset[preClass.item()]

