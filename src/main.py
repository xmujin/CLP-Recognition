# -*- coding: utf-8 -*-
# @Time    : 日期
# @Author  : 作者
# @Sid     : 学号
# @File    : 当前文件名
# @Description : 该程序的功能
import sys
from PyQt5.QtWidgets import QApplication
#from src.ui.my_mainwindow import MyMainWindow
import subprocess
import importlib
if __name__ == '__main__':
    command = r'H:\Anaconda3\envs\course_design\Scripts\pyuic5.exe src\ui\my_ui_mainwindow.ui -o src\ui\my_ui_mainwindow.py'
    # 执行命令
    subprocess.run(command, shell=True)
    module_name = "src.ui.my_mainwindow"
    myModule = importlib.import_module(module_name)

    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = myModule.MyMainWindow()
    #myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())  # 结束进程，退出程序


