from keras.models import load_model

import detection
import module_training
from detection import static_predict_expression

if __name__ == '__main__':
    '''
        train model part
    '''
    # model = module_selection.cnn_model()
    # history, history_predict = module_selection.train_model(model, 50)
    # module_selection.save_model(model)
    #
    # print(history)
    # print(history_predict)


    '''
        make prediction part
    '''
    expression_classifier = load_model("model/model.h5", compile=False)
    #
    # input size - 48*48
    # expression_target_size = expression_classifier.input_shape[1:3]

    # predict image
    # static_predict_expression("images/img_1.jpg", expression_classifier)

    '''
        real_time detection
    '''
    detection.real_time_detection(expression_classifier)