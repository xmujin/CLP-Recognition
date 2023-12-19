import cv2
import torch
import torchvision

from torchvision.transforms import functional
from test_character_recognition import CharCNN

dataset = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu',
           'zh_ji', 'zh_jin','zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong',
           'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun',
           'zh_zang', 'zh_zhe']


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

# boxes = prediction[0]['boxes'].cpu().numpy().astype(int)
