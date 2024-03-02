import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os

from save_read_files import load_data_from_jsons

TRAINING_CONFIG_JSON = './training_config.json'

sc = StandardScaler()


def env_setup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    tf.config.threading.set_inter_op_parallelism_threads(16)


def convert_to_decibels(data):
    return [10 * np.log10(linear_values) for linear_values in data]


def scaling_Xs(X_train, X_test):
    sc.fit(X_train)
    return sc.transform(X_train), sc.transform(X_test)


def divide_input_data(data):
    try:
        random.shuffle(data)
        X_data = np.array([data_object.reflectance for data_object in data])
        y_data = np.array([[data_object.n_eff, data_object.delta_n_eff, data_object.period, data_object.X_z]
                           for data_object in data])
        del data
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=4280)
        del X_data
        del y_data

        y_train[:, 0, :] = np.round(y_train[:, 0, :], 3)
        y_train[:, 1, :] = np.round(y_train[:, 1, :] * 1e3, 3)
        y_train[:, 2, :] = np.round(y_train[:, 2, :] * 1e6, 3)
        y_train[:, 3, :] = np.round(y_train[:, 3, :], 3)

        y_test[:, 0, :] = np.round(y_test[:, 0, :], 3)
        y_test[:, 1, :] = np.round(y_test[:, 1, :] * 1e3, 3)
        y_test[:, 2, :] = np.round(y_test[:, 2, :] * 1e6, 3)
        y_test[:, 3, :] = np.round(y_test[:, 3, :], 3)

        return X_train, X_test, y_train, y_test
    except Exception:
        print("Divide input data error")


def data_setup():
    try:
        data = load_data_from_jsons()
        (X_train, X_test, y_train, y_test) = divide_input_data(data)
        #(X_train_scaled, X_test_scaled) = scaling_Xs(X_train, X_test)


        X_train_reshaped = X_train.reshape(-1, 1, 500)
        X_test_reshaped = X_test.reshape(-1, 1, 500)
        del X_train
        # del X_train_scaled
        # del X_test_scaled
        del data
        return X_test, X_train_reshaped, X_test_reshaped, y_train, y_test
    except Exception:
        print("Data setup error")
