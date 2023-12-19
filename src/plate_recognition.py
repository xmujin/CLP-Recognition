# -*- coding: utf-8 -*-
# @Time    : 2023-12-14
# @Author  :
# @Sid     :
# @File    : plate_recognition.py
# @Description : 车辆定位深度学习数据集的建立，训练模型的建立
import os
import torch
import torchvision
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from show_label import GetBoxPoints


# 定义数据集类
class ClpDataset(Dataset):
    def __init__(self, data_path=None, transforms=None):
        self.data_dir = data_path
        # 获取每张图片的文件名
        self.image_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图像
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        # box = [50, 50, 200, 200]  # 边界框坐标，在这里要传入具体的值 [x_min, y_min, x_max, y_max]
        box = GetBoxPoints(self.image_files[idx])
        box = [item for sublist in box for item in sublist]
        print(box)
        label = 1  # 模拟标签（1 表示车牌）
        # 构建标注信息
        target = {"boxes": [box], "labels": [label]}
        # 数据转换
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


# 自定义数据集中样本的拼接函数
def collate_fn(batch):
    return tuple(zip(*batch))

# 定义数据转换
def transform(img, target):
    img = F.to_tensor(img)
    return img, target






# 训练模型
def train_model(model, dataloader, num_epochs=5, device='cuda'):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    for epoch in range(num_epochs):
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: torch.tensor(v).to(device) for k, v in target.items()} for target in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

if __name__ == "__main__":
    # 创建数据集和数据加载器
    # data_dir = '../dataset/CCPD2019/ccpd_base'
    data_dir = '../dataset/test'
    dataset = ClpDataset(data_dir, transforms=transform)
    # batch_size，每个批次训练的样本数
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 定义 Faster R-CNN 模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    num_classes = 2  # 背景和车牌两个类别
    # 这一行代码获取了 Faster R-CNN 模型中用于分类的全连接层（fc）的输入特征数。这个特征数会在下一步用于定义新的分类层。
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    ##
    # 这一行代码替换了模型的分类头部。
    # 它创建了一个新的 FastRCNNPredictor 对象，
    # 该对象使用之前获取的输入特征数 in_features
    # 和指定的类别数 num_classes 来定义一个新的分类层。
    # 这是为了适应你的特定任务，其中有两个类别。
    # #
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    print(torch.cuda.is_available())  # 查看CUDA是否可用
    print(torch.cuda.device_count())  # 查看可用的CUDA数量
    print(torch.version.cuda)  # 查看CUDA的版本号
    # 将模型移动到 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 训练模型
    train_model(model, dataloader, num_epochs=5, device=device)

    # 保存训练好的模型
    torch.save(model.state_dict(), '../model/car_plate_model.pth')
