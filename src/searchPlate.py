# -*- coding: UTF-8 -*-
import argparse
import os
import cv2
import torch
import copy
import numpy as np
from models.experimental import tryLoading
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords

clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
danger = ['危', '险']


def order_points(pts):  # 四个点按照左上 右上 右下 左下排列
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    print(rect)
    return rect


# 透视变换得到车牌小图
def perspectiveTransforms(image, pts):
    # rect = order_points(pts)
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# 加载检测模型
def loadModel(weights, device):
    try:
        model = tryLoading(weights, map_location=device)  # load FP32 model
    except Exception as e:
        print("sb", e, "sb")

    return model


# 拿到返回坐标中的四个顶点坐标
def convertFourPointCoordinates(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]
    coords[:, [1, 3, 5, 7]] -= pad[1]
    coords[:, :8] /= gain

    # 四点x，y坐标
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    coords[:, 4].clamp_(0, img0_shape[1])
    coords[:, 5].clamp_(0, img0_shape[0])
    coords[:, 6].clamp_(0, img0_shape[1])
    coords[:, 7].clamp_(0, img0_shape[0])

    print(coords)
    return coords


# 我们将图片传入模型之前，对图片进行了拉伸变换。因此我们得到的车牌坐标不是原图的，需要进行转换才可以。
def get_plate_rec_landmark(img, landmarks):
    landmarks_np = np.zeros((4, 2))
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i] = np.array([point_x, point_y])

    roi_img = perspectiveTransforms(img, landmarks_np)

    return roi_img


# 获取车牌信息
def detect_Recognition_plate(model, orgimg, device, img_size):
    # 得分阈值
    scoreThreshold = 0.3

    # nms的iou值
    iouThreshold = 0.5

    img0 = copy.deepcopy(orgimg)

    assert orgimg is not None, 'Image Not Found '
    h0, w0 = orgimg.shape[:2]
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    # 检测前处理，图片长宽变为32倍数，比如变为640X640
    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416  图片的BGR排列转为RGB,然后将图片的H,W,C排列变为C,H,W排列

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, scoreThreshold, iouThreshold)

    # 检测过程
    for i, det in enumerate(pred):  # 检测每一个车牌，在这里我们只检测到一个车牌即可返回结果
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # 拿到det中的车牌四个顶点坐标，这四个顶点坐标不是原图车牌中的车牌顶点坐标。需要后续变换
            det[:, 5:13] = convertFourPointCoordinates(img.shape[2:], det[:, 5:13], orgimg.shape).round()

            for j in range(det.size()[0]):
                landmarks = det[j, 5:13].view(-1).tolist()
                resultPicture = get_plate_rec_landmark(orgimg, landmarks)
                return resultPicture

    return None


global detect_model
detect_model = None

global device
device = None

global img_size
img_size = None


# 车牌定位检测前初始化模型等
def plate_recognition_init():
    global detect_model
    global device
    global img_size

    # 默认使用GPU来加载，如果GPU不可用，那么就用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = 'result'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    try:
        detect_model = loadModel('weights/plate_detect.pt', device)  # 初始化检测模型
    except Exception as E:
        print("sdf", E, "sdf")
    detect_model, device, img_size = detect_model, device, 640

    return detect_model, device, img_size


def getPicture(img):
    """
    传入原图，返回分割好图片的原图
    :param img:
    :return:
    """
    if detect_model is None:
        plate_recognition_init()

    if img.shape[-1] == 4:  # 图片如果是4个通道的，将其转为3个通道
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 分割车牌并进行，倾斜校正后的彩色图
    correctedOriginal = detect_Recognition_plate(detect_model, img, device, img_size)  # 检测以及识别车牌
    return correctedOriginal
if __name__ == "__main__":
    img = cv2.imread('../other/VLP-Recognition/src/hqh_task/imgs/1.jpg')
    # cv2.imshow('1', img)
    img = getPicture(img)
    cv2.imshow('1', img)
    cv2.waitKey(0)
