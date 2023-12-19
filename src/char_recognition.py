# -*- coding: utf-8 -*-
# @Time    : 2023-12-18
# @Author  : 向波
# @Sid     : 12103990437
# @File    : char_recognition.py
# @Description :使用字符模型，对分割好的字符进行识别
import cv2
import torch
import torchvision

from torchvision.transforms import functional
from build_char_recognition_model import CharCNN

dataset = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵', '桂', '黑', '沪',
           '冀', '津','京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼',
           '陕', '苏', '晋', '皖', '湘', '新', '豫', '渝', '粤', '云',
           '藏', '浙']



num_classes = 67  # 背景和车牌两个类别
model = CharCNN(num_classes)
# 加载模型参数
model.load_state_dict(torch.load('../model/plate_char_model.pth'))
device = torch.device('cuda')
model.to(device)
# 读取图片
img = cv2.imread("../images/cnn_char_train/0/4-3.jpg", cv2.IMREAD_GRAYSCALE)

# 将图片转换为张量, unsqueeze表示在第0维前插入一个维度，该张量原来维度为1,20,20，为通道数，高，宽
# 变换后为1, 1, 20, 20, 最前面的1相当与批次，此处就一个图片。
img = functional.to_tensor(img).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    prediction = model(img)

preClass = torch.argmax(prediction)
print("该字符为:", dataset[preClass.item()])
