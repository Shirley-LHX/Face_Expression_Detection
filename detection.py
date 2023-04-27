import time

import cv2

import utils
from utils import load_image, get_coordinates, preprocess_input, draw_detection_res
import numpy as np

SAVE_IMG_PATH = "/Users/lhx/PycharmProjects/Face_Expression_Detection/images/result/{}.jpg"

# load face detection model
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX


def static_predict_expression(img_path, model):
    rgb_image = load_image(img_path, grayscale=False)
    gray_image = load_image(img_path, grayscale=True)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')
    target_size = model.input_shape[1:3]
    result_map = []

    faces = face_detection.detectMultiScale(gray_image, 1.3, 5)
    for face_coordinates in faces:
        x1, x2, y1, y2 = get_coordinates(face_coordinates)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, target_size)
        except:
            print("Convert Failed")
            continue

        # normalization the image - (1, 48, 48, 1)
        gray_face = preprocess_input(gray_face)
        emotion_text = utils.make_prediction(model, gray_face)
        print('emotion_text = ', emotion_text)
        result_map.append(emotion_text)

        # visualization
        draw_detection_res(face_coordinates, rgb_image, emotion_text, 1, 2)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    img_path = SAVE_IMG_PATH.format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    cv2.imwrite(img_path, bgr_image)
    return img_path, result_map


def dynamic_monitor(model, frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # recognize faces
    face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_detection.detectMultiScale(gray_img, 1.3, 5)
    for (x, y, w, h) in faces:
        fc = gray_img[y:y + h, x:x + w]
        roi = cv2.resize(fc, (48, 48))
        pred = utils.make_prediction(model, roi[np.newaxis, :, :, np.newaxis])

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 51), 2)
        cv2.putText(frame, pred, (x, y),cv2.FONT_HERSHEY_SIMPLEX, 2, (44, 44, 238), 3, cv2.LINE_AA)

    return frame


def real_time_detection(model):
    # 打开内置摄像头
    video = cv2.VideoCapture(0)
    # 设置视频窗口大小
    video.set(3, 640)
    video.set(4, 480)
    faceNum = 0

    while True:
        # 读取视频帧
        ret, frame = video.read()
        # 图像灰度处理
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 设定人脸识别参数
        faces = face_detection.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_frame[y:y + h, x:x + w]

            roi = cv2.resize(fc, (48, 48))
            print(roi)
            pred = utils.make_prediction(model, roi[np.newaxis, :, :, np.newaxis])
            print(pred)

            cv2.putText(frame, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 显示图像
        cv2.imshow('Output', frame)

        # L键退出显示
        if cv2.waitKey(10) & 0xff == ord('e'):
            break
    # 释放资源
    cv2.destroyAllWindows()
    video.release()
