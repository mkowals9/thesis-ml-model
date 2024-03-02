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
    sc.fit(X_train)
    return sc.transform(X_train), sc.transform(X_test)


def divide_input_data(data):
    try:
        random.shuffle(data)
        X_data = [data_object["reflectance"] for data_object in data]
        y_data = [[data_object["n_eff"], data_object["delta_n_eff"], data_object["period"], data_object["X_z"]]
                  for data_object in data]
        del data
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=4280)
        del X_data
        del y_data

        # Preprocessing y_train and y_test
        def preprocess_data(data):
            return [[round(sublist[0], 3), sublist[1] * 1e3, sublist[2] * 1e6, round(sublist[3], 2)] for sublist in
                    data]

        y_train = preprocess_data(y_train)
        y_test = preprocess_data(y_test)

        # Another round of preprocessing
        def round_data(data):
            return [[sublist[0], round(sublist[1], 3), round(sublist[2], 3), sublist[3]] for sublist in data]

        y_train = round_data(y_train)
        y_test = round_data(y_test)

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        return X_train, X_test, y_train, y_test
    except Exception:
        print("Divide input data error")


def data_setup():
    try:
        data = load_data_from_jsons()
        (X_train, X_test, y_train, y_test) = divide_input_data(data)
        (X_train_scaled, X_test_scaled) = scaling_Xs(X_train, X_test)

        X_train_reshaped = X_train_scaled.reshape(len(X_train_scaled), -1)
        X_test_reshaped = X_test_scaled.reshape(len(X_test_scaled), -1)
        del X_train_scaled
        del X_test_scaled
        return X_test, X_train_reshaped, X_test_reshaped, y_train, y_test
    except Exception:
        print("Data setup error")
