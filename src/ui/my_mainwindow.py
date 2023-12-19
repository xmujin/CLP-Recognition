import sys

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QDir, QStringListModel, QModelIndex
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog

from src.ui.my_ui_mainwindow import Ui_MainWindow
from src.show_label import GetVertexes
from src.plate_right import GetRightPlate
from src.character_split import FindVSplitPos, FindHSplitPos, SplitCharacters
from src.char_recognition import Get



class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)# parent确定父组件和子组件的关系，与继承无关
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类
        #加载字符检测模型
        self.model = GetCharModel()

    def showFileDialog(self):
        # 显示文件夹选择对话框
        path = QFileDialog.getExistingDirectory(self, '选择文件夹', '.', QFileDialog.ShowDirsOnly)
        self.lineEdit.setText(path)
        dir = QDir(path)
        fileList = [item.fileName() for item in dir.entryInfoList() if item.isFile()]
        model = QStringListModel()
        model.setStringList(fileList)
        self.listView.setModel(model)
    def showPicture(self, index: QModelIndex):
        self.fileName = index.data()
        #print(self.fileName)
        self.filePath = QDir(self.lineEdit.text()).filePath(self.fileName)
        # 显示图片到 QLabel 上
        # pixmap = QPixmap(file_path)
        pixmap = QPixmap(self.filePath)
        #self.picture.setScaledContents(True)
        self.picture.setPixmap(pixmap)
        # self.picture.set
        #self.picture.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # 获取 QLabel 的大小
        label_size = self.picture.size()
        #print(label_size)
        # 调整图片大小以适应 QLabel
        resized_pixmap = pixmap.scaled(label_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        # 显示调整大小后的图片到 QLabel 上
        self.picture.setPixmap(resized_pixmap)
        #file_name = self.listView.model().data(index)
        #print(file_name)
        # file_list = self.getFileList(path)
        # self.listView.setStringList(file_list)
    # def getFileList(self, path):
    #     # 使用 QDir 获取文件夹下的文件列表
    #     dir_model = QDir(path)
    #     file_list = [file_info.fileName() for file_info in dir_model.entryInfoList() if file_info.isFile()]
    #     return file_list
    def cv2BgrToPixmap(self, img, size):
        cur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # todo 图片处理
        _, cur = cv2.threshold(cur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #
        height, width = cur.shape
        bytes_per_line = 1 * width
        qt_image = QImage(cur.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(size, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        return pixmap
    def cv2BinaryToPixmap(self, cur, size):
        cur = cv2.cvtColor(cur, cv2.COLOR_GRAY2RGB)
        height, width, h = cur.shape
        bytes_per_line = h * width
        qt_image = QImage(cur.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(size, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        return pixmap

    def showRecognition(self):
        img = cv2.imread(self.filePath)
        # 该车牌
        img2 = GetRightPlate(img, self.fileName)
        # nextImg，单通道灰度图
        nextImg = img2.copy()
        nextImg = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
        _, nextImg = cv2.threshold(nextImg, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(nextImg)
        # splitPos =

        characters = SplitCharacters(nextImg)
        print(len(characters))
        vertexs = GetVertexes(self.fileName)
        vertexsT = np.array([vertexs[0], vertexs[1], vertexs[3], vertexs[2]], dtype=int)
        cv2.polylines(img, [vertexsT], isClosed=True, color=(0, 255, 0), thickness=2)
        self.picture.setPixmap(self.cv2BgrToPixmap(img, self.picture.size()))
        self.platePos.setPixmap(self.cv2BgrToPixmap(img2, self.platePos.size()))
        # 将分割好的字符进行显示
        #print(characters.shape)
        # print(characters[0].shape)
        try:
            self.char_a.setPixmap(self.cv2BinaryToPixmap(characters[0], self.char_a.size()))
            self.char_b.setPixmap(self.cv2BinaryToPixmap(characters[1], self.char_b.size()))
            self.char_c.setPixmap(self.cv2BinaryToPixmap(characters[2], self.char_c.size()))
            self.char_d.setPixmap(self.cv2BinaryToPixmap(characters[3], self.char_d.size()))
            self.char_e.setPixmap(self.cv2BinaryToPixmap(characters[4], self.char_e.size()))
            self.char_f.setPixmap(self.cv2BinaryToPixmap(characters[5], self.char_f.size()))
            self.char_g.setPixmap(self.cv2BinaryToPixmap(characters[6], self.char_g.size()))
        except IndexError:
            print("sb")

        # 将分割好的的字符进行识别



        pass