# -*- coding: utf-8 -*-
# @Time    : 2023-12-14
# @Author  : 向波
# @Sid     : 12103990437
# @File    : character.py
# @Description : 用于字符识别
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# 假设你的字符数据集是一个自定义的数据集类，继承自 torch.utils.data.Dataset
# 你需要适应性调整数据预处理部分以适应 ResNet 的输入要求
# 34
# 31
charList = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新",
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]


# 假设字符图片文件名如下格式0_1.jpg，其中第一个0代表该字符是皖, 第二个表示这是该字符对应的第一个图片
def GetLabels(imgFileName): # 传入文件名，如0_1.jpg
    fileBaseName = os.path.splitext(imgFileName)[0]
    label, _ = fileBaseName.split('-')
    return label

class CharacterDataset(Dataset):
    def __init__(self, data_path=None, wordList=None):
        self.data_dir = data_path
        self.wordList = wordList
        self.image_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.labels = [wordList[GetLabels(img)] for img in self.image_files]



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        # 在这里，你可以使用图像处理库（例如PIL或OpenCV）加载图像数据
        # 以及将字符标签转换为模型可以处理的形式（例如对字符进行索引编码）
        img = Image.open(image_path)
        label_index = self.wordList.index(label)
        return img, label_index




# 定义训练循环
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # 正向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 打印每个epoch的平均损失
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")



if __name__ == "__main__":
    # 创建数据集和数据加载器
    # data_dir = '../dataset/CCPD2019/ccpd_base'
    data_dir = '../dataset/last'
    dataset = CharacterDataset(data_dir, charList)
    # batch_size，每个批次训练的样本数
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # 加载预训练的 ResNet 模型，去掉最后一层（全连接层）
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])

    # 替换最后一层，适应你的字符识别任务
    num_classes = 65  # 假设有10个字符类别
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # 定义损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 设置设备（CPU或GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 使用训练函数进行训练
    train_model(model, dataloader, criterion, optimizer, num_epochs=10, device=device)

    # 保存训练好的模型
    torch.save(model.state_dict(), 'car_plate_model.pth')
