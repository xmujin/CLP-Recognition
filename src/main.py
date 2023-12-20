# -*- coding: utf-8 -*-
# @Time    : 2023-12-17
# @Author  : 向波
# @Sid     : 12103990437
# @File    : main.py
# @Description : 主程序入口
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

