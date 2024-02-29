import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
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
    return (np.array([sc.fit_transform(X) for X in X_train]),
            np.array([sc.fit_transform(X) for X in X_test]))


def divide_input_data(data):
    pass


def data_setup():
    try:
        data = load_data_from_jsons()
        (X_train, X_test, y_train, y_test) = divide_input_data(data)
        (X_train_scaled, X_test_scaled) = scaling_Xs(X_train, X_test)

        X_train_reshaped = X_train_scaled.reshape(len(X_train_scaled), -1)
        X_test_reshaped = X_test_scaled.reshape(len(X_test_scaled), -1)
        return X_test, X_train_reshaped, X_test_reshaped, y_train, y_test
    except Exception:
        print("Data setup error")
