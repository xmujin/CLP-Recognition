import sys

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QDir, QStringListModel, QModelIndex, QThread, pyqtSignal, QItemSelectionModel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel
from src.ui.my_ui_mainwindow import Ui_MainWindow
from src.show_label import GetVertexes
from src.plate_right import GetRightPlate
from src.character_split import FindVSplitPos, FindHSplitPos, SplitCharacters, SplitPosPlate
from src.char_recognition import GetCharModel, CharRecModel
from src.searchPlate import getPicture

class ModelLoaderThread(QThread):
    modelSig = pyqtSignal(object)

    def run(self):
        self.modelSig.emit(CharRecModel("model/plate_char_model.pth"))
        print("模型加载完成")


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)  # parent确定父组件和子组件的关系，与继承无关
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类
        # 加载字符检测模型
        # self.char_rec_mode = CharRecModel("model/plate_char_model.pth")
        self.char_rec_mode = None
        self.model_loader_thread = ModelLoaderThread()
        self.model_loader_thread.modelSig.connect(self.GetLoadedModel)
        # 启动模型加载线程
        self.model_loader_thread.start()
        self.labels = [self.char_a, self.char_b, self.char_c,
                       self.char_d, self.char_e, self.char_f, self.char_g]
        self.reses = [self.res_a, self.res_b, self.res_c,
                      self.res_d, self.res_e, self.res_f, self.res_g]
        # 定义菜单选择项
        self.selectedMenuIdx = 0 # 默认是一类车牌检测
        self.action1.triggered.connect(self.ChangeMenuIdx0)
        self.action2.triggered.connect(self.ChangeMenuIdx1)


    def ChangeMenuIdx0(self):
        self.selectedMenuIdx = 0
        theLabel = QLabel("第I类集中测试", self)
        self.statusbar.addWidget(theLabel)
    def ChangeMenuIdx1(self):
        self.selectedMenuIdx = 1
        theLabel = QLabel("第II类集中测试", self)
        self.statusbar.addWidget(theLabel)

    def GetLoadedModel(self, model):
        self.char_rec_mode = model

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
        # print(self.fileName)
        self.filePath = QDir(self.lineEdit.text()).filePath(self.fileName)
        # 显示图片到 QLabel 上
        # pixmap = QPixmap(file_path)
        pixmap = QPixmap(self.filePath)
        # self.picture.setScaledContents(True)
        self.picture.setPixmap(pixmap)
        # self.picture.set
        # self.picture.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # 获取 QLabel 的大小
        label_size = self.picture.size()
        # print(label_size)
        # 调整图片大小以适应 QLabel
        resized_pixmap = pixmap.scaled(label_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        # 显示调整大小后的图片到 QLabel 上
        self.picture.setPixmap(resized_pixmap)
        # file_name = self.listView.model().data(index)
        # print(file_name)
        # file_list = self.getFileList(path)
        # self.listView.setStringList(file_list)

    # def getFileList(self, path):
    #     # 使用 QDir 获取文件夹下的文件列表
    #     dir_model = QDir(path)
    #     file_list = [file_info.fileName() for file_info in dir_model.entryInfoList() if file_info.isFile()]
    #     return file_list
    def cv2BgrToPixmap(self, img, size):
        #cur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cur = img.copy()
        # todo 图片处理
        # _, cur = cv2.threshold(cur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        height, width, channel = cur.shape
        bytes_per_line = channel * width
        #qt_image = QImage(cur.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        qt_image = QImage(cur.data, width, height, bytes_per_line, QImage.Format_BGR888)
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
        self.showRecognition1()
    def showRecognition1(self):
        img = cv2.imread(self.filePath)
        imgHaveLine = img.copy()
        if self.selectedMenuIdx == 0:
            # 该车牌为矫正后的车牌, 为BGR图
            imgRight = GetRightPlate(img, self.fileName)
        else:
            # 对于二类识别，车牌图像不应由文件名给出
            imgRight = getPicture(img)
        # charIm 用于分割的图，需要转为二值图，从已经矫正的车牌图的副本
        charImg = imgRight.copy()
        charImg = cv2.cvtColor(charImg, cv2.COLOR_BGR2GRAY)
        _, charImg = cv2.threshold(charImg, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 具有分割位置的图,需要进行处理
        splitPlateImg = SplitPosPlate(imgRight)

        characters = SplitCharacters(charImg)

        if self.selectedMenuIdx == 0:
            vertexs = GetVertexes(self.fileName)
            vertexsT = np.array([vertexs[0], vertexs[1], vertexs[3], vertexs[2]], dtype=int)
            cv2.polylines(imgHaveLine, [vertexsT], isClosed=True, color=(0, 255, 0), thickness=2)
            self.picture.setPixmap(self.cv2BgrToPixmap(imgHaveLine, self.picture.size()))

        # 显示只有车牌的图片
        self.platePos.setPixmap(self.cv2BgrToPixmap(imgRight, self.platePos.size()))

        # 显示具有分割位置的车牌的图片：
        self.splitPosPlate.setPixmap(self.cv2BgrToPixmap(splitPlateImg, self.splitPosPlate.size()))

        # 将分割好的字符进行显示
        # print(characters.shape)
        # print(characters[0].shape)
        try:
            for label in self.labels:
                label.clear()
            self.char_a.setPixmap(self.cv2BinaryToPixmap(characters[0], self.char_a.size()))
            self.char_b.setPixmap(self.cv2BinaryToPixmap(characters[1], self.char_b.size()))
            self.char_c.setPixmap(self.cv2BinaryToPixmap(characters[2], self.char_c.size()))
            self.char_d.setPixmap(self.cv2BinaryToPixmap(characters[3], self.char_d.size()))
            self.char_e.setPixmap(self.cv2BinaryToPixmap(characters[4], self.char_e.size()))
            self.char_f.setPixmap(self.cv2BinaryToPixmap(characters[5], self.char_f.size()))
            self.char_g.setPixmap(self.cv2BinaryToPixmap(characters[6], self.char_g.size()))
        except IndexError:
            pass

        print("开始识别")
        # 将分割好的的字符进行识别
        res = ""
        for index, char in enumerate(characters):
            res += self.char_rec_mode.GetChar(char, index)
        try:
            for item in self.reses:
                item.clear()
            self.res_a.setText(res[0])
            self.res_b.setText(res[1])
            self.res_c.setText(res[2])
            self.res_d.setText(res[3])
            self.res_e.setText(res[4])
            self.res_f.setText(res[5])
            self.res_g.setText(res[6])
        except Exception as e:
            pass
        last = ""
        for i in range(len(res)):
            last += res[i]
            if i == 1:
                last += '·'
        self.result.setText(last)
        pass






    def toPre(self):
        # 获取当前所在项的索引，只用到了行row
        curIdx = self.listView.selectionModel().selectedIndexes()[0]
        preIdx = self.listView.model().index(curIdx.row() - 1, 0)
        if preIdx.isValid():
            self.listView.selectionModel().select(preIdx,
            QItemSelectionModel.SelectionFlag.Select |
            QItemSelectionModel.SelectionFlag.SelectCurrent)
            self.showPicture(preIdx)
    def toNext(self):
        curIdx = self.listView.selectionModel().selectedIndexes()[0]
        nextIdx = self.listView.model().index(curIdx.row() + 1, 0)
        if nextIdx.isValid():
            self.listView.selectionModel().select(nextIdx,
            QItemSelectionModel.SelectionFlag.Select |
            QItemSelectionModel.SelectionFlag.SelectCurrent)
            self.showPicture(nextIdx)

