# -*- coding: utf-8 -*-
# @Time    : 2023-12-14
# @Author  :
# @Sid     :
# @File    : test_model.py
# @Description : 测试定位模型

import torch
from PIL import Image, ImageDraw
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from plate_recognition import ClpDataset
from torchvision.transforms import functional as F
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
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
# 将训练好的参数加载到模型中
model.load_state_dict(torch.load('car_plate_model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 在示例图像上测试模型
sample_image = Image.open('../pic/test/hhh.jpg').convert('RGB')
sb = sample_image.copy()
# 将该图片转化为张量
sample_image = F.to_tensor(sample_image).unsqueeze(0).to(device)
draw = ImageDraw.Draw(sb)
model.eval()
with torch.no_grad():
    prediction = model(sample_image)

# 在图像上绘制边界框


boxes = prediction[0]['boxes'].cpu().numpy().astype(int)
print(boxes)
print(tuple(*boxes))
# draw.rectangle([227,352,512,451], outline="#00FF00", width=3)
draw.rectangle(tuple(*boxes), outline="#00FF00", width=3)

# 保存或显示图像
# sample_image = F.to_pil_image(sample_image.squeeze(0).cpu())
sb.show()
