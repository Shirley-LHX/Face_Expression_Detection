import sys
import time

import qimage2ndarray
from keras.models import load_model
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer, QCoreApplication
import cv2

import detection
from designer_ui import mainWin

MODEL_PATH = "../model/{}.h5"


class mainWindow(mainWin.Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()
        self.setupUi(self)
        self.prepParameters()
        self.prepCallBack()

    def prepParameters(self):
        self.stackedWidget.setCurrentIndex(1)
        # 通过计时器实现对摄像头图像的循环读取和显示
        self.Timer = QTimer()
        self.camstopBt.setEnabled(False)
        self.camera = None
        self.imgPath = ""
        # self.FilePathLE.setText(self.RecordPath)将路径名显示在文本框FilePathLE中
        # self.FilePathLE.setText(self.RecordPath)
        # self.Image_num定义读取图片的次数
        # 初始化识别model！
        self.model_path = MODEL_PATH.format(self.comboBox.currentText())
        self.model = load_model(self.model_path, compile=False)

    def prepCallBack(self):
        # 系统主要功能
        self.comboBox.currentTextChanged.connect(self.change_model)
        self.picBt.clicked.connect(self.show_static_detection)
        self.videoBt.clicked.connect(self.real_time_detection)
        self.exitBt.clicked.connect(self.exitApp)
        # 静态识别图片主要功能
        self.fileBt.clicked.connect(self.choice_image)
        self.startBt.clicked.connect(self.static_detection)
        # 摄像头实时监控动态检测主要功能
        self.Timer.timeout.connect(self.dynamic_detection)
        self.startBt_2.clicked.connect(self.startCamera)
        self.camstopBt.clicked.connect(self.stopCamera)



    ''' static detect part '''
    def choice_image(self):
        self.imgPath = \
        QFileDialog.getOpenFileName(self, "选取文件", "/Users/lhx/PycharmProjects/Face_Expression_Detection/images",
                                    "*.jpg;*.png")[0]
        print(self.imgPath)
        # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
        img = QPixmap(self.imgPath)
        self.imageShow.setScaledContents(True)
        # 在label控件上显示选择的图片
        self.imageShow.setPixmap(img)

    def static_detection(self):
        if self.imgPath == "":
            QMessageBox.warning(self, 'warning', "Please choose a image!", buttons=QMessageBox.Ok)
        else:
            new_imgPath, resultMap = detection.static_predict_expression(self.imgPath, self.model)
            img = QPixmap(new_imgPath)
            self.imageShow.setScaledContents(True)
            self.imageShow.setPixmap(img)
            self.label_3.setText(str(resultMap))


    ''' system related module '''
    def show_static_detection(self):
        if self.stackedWidget.currentIndex() == 2 & self.camstopBt.isEnabled():
            QMessageBox.warning(self, "Warning", "You can't change the page now!", QMessageBox.Ok)
        if self.camera:
            self.camera.release()
        self.stackedWidget.setCurrentIndex(1)

    def real_time_detection(self):
        self.prepCamera()
        self.stackedWidget.setCurrentIndex(2)

    def change_model(self):
        self.model_path = MODEL_PATH.format(self.comboBox.currentText())
        self.model = load_model(self.model_path, compile=False)

    def exitApp(self):
        self.Timer.stop()
        if self.camera:
            self.camera.release()
        QCoreApplication.quit()


    ''' Real-time detection'''
    # 初始化摄像头
    def prepCamera(self):
        try:
            # 打开摄像头
            self.camera = cv2.VideoCapture(0)
            self.MsgTE.clear()  # 清空文本框MsgTE中的内容
            self.MsgTE.append('Oboard camera connected...')
        except Exception as e:
            self.MsgTE.clear()
            self.MsgTE.append(str(e))

    def startCamera(self):
        self.prepCamera()
        self.MsgTE.append("Starting real-time monitoring...")
        self.startBt_2.setEnabled(False)
        self.camstopBt.setEnabled(True)
        self.Timer.start(1)  # self.Timer.start(1)用来启动计时器 -1ms启动一次计时器

    def dynamic_detection(self):
        success, self.v_image = self.camera.read()  # 从视频流中读取
        if success:
            frame = detection.dynamic_monitor(self.model, self.v_image)
            # keep origin color
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 视频流的长和宽
            height, width = rgb_img.shape[:2]
            pixmap = QImage(rgb_img, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
            ratio = max(width / self.cameraShow.width(), height / self.cameraShow.height())
            pixmap.setDevicePixelRatio(ratio)
            # 视频流置于label中间部分播放
            self.cameraShow.setPixmap(pixmap)


    def stopCamera(self):
        self.MsgTE.append("Stop to real-time monitoring...")
        self.startBt_2.setEnabled(True)
        self.camstopBt.setEnabled(False)
        self.camera.release()
        self.startBt_2.clicked.connect(self.startCamera)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = mainWindow()
    main_window.setWindowTitle("Facial Expression Detection")
    main_window.show()
    sys.exit(app.exec())
