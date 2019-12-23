from sklearn.metrics import f1_score
import pickle
from humor_recognition.data import load_test
import pandas as pd
import numpy as np
from writer_json import read


def classifier_func(X_test, y_test, file_name):
    loaded_model = pickle.load(open(file_name, 'rb'))
    print(f1_score(y_test, loaded_model.predict(X_test)))


def class_for_NN(X_test, y_test, file_name):
    loaded_model = pickle.load(open(file_name, 'rb'))
    print(f1_score(y_test, loaded_model.predict_classes(X_test)))


if __name__ == '__main__':
    X_test = np.array(list(map(lambda elem: np.array(elem), read('X_test.json'))))
    y_test = pd.Series(load_test()).values
    """
    Модель с алгоритмом "случайного леса"
    """
    file_name = 'model_RFC.sav'
    classifier_func(X_test, y_test, file_name)

    """
    Нейросеть с тремя полносвязными слоями
    """
    file_name = 'model_NN.sav'
    class_for_NN(X_test, y_test, file_name)
    """
    Модель, основанная на логистической регрессии
    """
    file_name = 'finalized_model.sav'
    classifier_func(X_test, y_test, file_name)


