from keras.callbacks import ModelCheckpoint
import utils
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, \
    MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator


def cnn_model(n_class=7, learning_rate=0.001):
    """
        Create a convolution neural network
        在输入层后加一层的1*1卷积增加非线性表示
    """
    model = Sequential()

    # 1st - Convolution
    model.add(Conv2D(32, (1, 1), padding='same', input_shape=(48, 48, 1), activation='relu'))
    model.add(BatchNormalization())

    # 2nd - Convolution
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd Convolution layer
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 4th Convolution layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flattening
    model.add(Flatten())
    # Fully connected layer 1st layer
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_class, activation='softmax'))
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


# def resnet_model(self):
#     a = 1
#     return a

def train_model(model, epochs):
    # 以文件分类名划分label
    train_generator = utils.train_data_augment()
    val_generator = utils.test_data_process()
    eval_generator = utils.test_data_process("dataset/PublicTest/")

    checkpoint_new = ModelCheckpoint("model/model_weights_py.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
    callbacks_list_new = [checkpoint_new]

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_generator.n // train_generator.batch_size,
                                  epochs=epochs,
                                  validation_data=val_generator,
                                  validation_steps=val_generator.n // val_generator.batch_size,
                                  callbacks=callbacks_list_new
                                  )
    history_predict = model.predict_generator(eval_generator, steps=2000)  # 训练模型预测

    return history, history_predict


def save_model(model):
    # 使用model.to_json()函数只保存模型的结构，而不包含其权重及训练的配置（损失函数、优化器）
    model_json = model.to_json()
    with open("model/model_json.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model_weight.h5')
    model.save('model.h5')
    print('model saved')
