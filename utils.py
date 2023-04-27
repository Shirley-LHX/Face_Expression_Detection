import cv2
import numpy as np
from keras.preprocessing import image
import os
from keras_preprocessing.image import ImageDataGenerator

'''
    training data' tools
'''
ExpressionDict = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"

}


def train_data_augment(url="dataset/Training/", pic_size=48, batch_size=128, shear_range=0.2, zoom_range=0.2):
    """
    Data Augmentation and preprocess of training data
    """
    datagen_train = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=shear_range,
                                       zoom_range=zoom_range,
                                       horizontal_flip=True)

    train_generator = datagen_train.flow_from_directory(url, target_size=(pic_size, pic_size),
                                                        color_mode="grayscale", batch_size=batch_size,
                                                        class_mode='categorical', shuffle=True)
    return train_generator


def test_data_process(url="dataset/PrivateTest/", pic_size=48, batch_size=128):
    """
        image preprocess of validation and test data
    """
    datagen_validation = ImageDataGenerator(rescale=1. / 255)

    validation_generator = datagen_validation.flow_from_directory(url, target_size=(pic_size, pic_size),
                                                                  color_mode="grayscale", batch_size=batch_size,
                                                                  class_mode='categorical', shuffle=False)
    return validation_generator


'''
    detection data's tool
'''


def read_data(path="dataset/Training/", class_label=ExpressionDict):
    X = []
    Y = []
    for label in os.listdir(path):  # os.listdir用于返回指定的文件夹包含的文件或文件夹的名字的列表，此处遍历每个文件夹
        for img_file in os.listdir(os.path.join(path, label)):  # 遍历每个表情文件夹下的图像
            image = cv2.imread(os.path.join(path, label, img_file))  # 读取图像
            result = image / 255.0  # 图像归一化
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            X.append(result)  # 将读取到的所有图像的矩阵形式拼接在一起
            Y.append(class_label[label])  # 将读取到的所有图像的标签拼接在一起
    return X, Y


def load_image(image_path, grayscale=False, target_size=None):
    color_mode = 'grayscale'
    if grayscale == False:
        color_mode = 'rgb'
    else:
        grayscale = False
    pill_image = image.image_utils.load_img(image_path, grayscale, color_mode, target_size)
    return image.image_utils.img_to_array(pill_image)


def preprocess_input(data):
    x = np.array(data, dtype=np.float32)
    x = x / 255.0
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, -1)
    return x


def get_coordinates(face_coordinates):
    x, y, width, height = face_coordinates
    return x, x + width, y, y + height


def draw_detection_res(face_coordinates, image_array, text, font_scale=2, thickness=3):
    x, y, width, height = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + width, y + height), (51, 255, 255), 2)
    cv2.putText(image_array, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (238, 44, 44), thickness, cv2.LINE_AA)



def make_prediction(model, img):
    pred = model.predict(img)
    return ExpressionDict[np.argmax(pred)]
