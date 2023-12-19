import os

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional
from torchvision.transforms import functional
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda")

dataset = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu',
           'zh_ji', 'zh_jin','zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong',
           'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun',
           'zh_zang', 'zh_zhe']


# 定义数据集类
class CharDataset(Dataset):
    def __init__(self, root=None, dataset=None, transforms=None):
        self.root = root
        self.dataset = dataset
        self.transforms = transforms
        # 列出当前数据集的所有图像文件的相对路径
        self.fileNames = self.listAllFiles(self.root)
        self.labels = [i for i in range(len(self.dataset))]
    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        # 获取指定文件的相对路径
        fileName = self.fileNames[idx]
        srcImg = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
        #resizeImg = cv2.resize(srcImg, (20, 20))
        # 获取分类名
        className = os.path.split(os.path.dirname(fileName))[-1]
        # 获取分类名的索引
        label = self.dataset.index(className)
        if self.transforms is not None:
            img, label = self.transforms(srcImg, label)
        return img, label

    def listAllFiles(self, root):
        fileNames = []
        list = os.listdir(root)
        for i in range(len(list)):
            element = os.path.join(root, list[i])
            if os.path.isdir(element):
                temp_dir = os.path.split(element)[-1]
                if temp_dir in self.dataset:
                    fileNames.extend(self.listAllFiles(element))
            elif os.path.isfile(element):
                fileNames.append(element)
        return fileNames


# 定义CNN模型
class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #x = x.view(-1, 128 * 8 * 8)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# 定义数据变换
def transforms(srcimg, label):
    img = cv2.resize(srcimg, (20, 20))
    img = functional.to_tensor(img)
    return img, label

# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            print(inputs.shape)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

if __name__ == "__main__":
    root = "../images/cnn_char_train"
    char_dataset = CharDataset(root, dataset, transforms=transforms)
    train_loader = DataLoader(char_dataset, batch_size=64, shuffle=True, drop_last=True)
    #
    # # 创建模型
    num_classes = 67  # 67种字符
    model = CharCNN(num_classes)
    model.to(device)
    # #
    # # # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # #
    # # # 训练模型
    train(model, train_loader, criterion, optimizer, num_epochs=10)
    # 保存训练好的模型
    torch.save(model.state_dict(), "../model/plate_char_model.pth")
