import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import scipy

from save_read_files import load_data_from_jsons, load_chunked_data_npy

TRAINING_CONFIG_JSON = './training_config.json'

sc = StandardScaler()


def env_setup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ["XDG_SESSION_TYPE"] = "xcb"

    # tf.config.threading.set_inter_op_parallelism_threads(16)


def convert_to_decibels(data):
    return np.array([math.log10(x) * 10 for x in data])


def scaling_Xs_only_reflectance(X_train, X_test):
    sc.fit(X_train)
    return sc.transform(X_train), sc.transform(X_test)


def scaling_Xs_wavelength_reflectance(X_train, X_test):
    scaler = StandardScaler()

    # Flatten and concatenate the training data for fitting the scaler
    X_train_concatenated = np.concatenate([X.flatten() for X in X_train]).reshape(-1, 1)
    scaler.fit(X_train_concatenated)

    # Transform both training and testing data using the fitted scaler
    X_train_scaled = np.array([scaler.transform(X.flatten().reshape(-1, 1)).reshape(X.shape) for X in X_train])
    X_test_scaled = np.array([scaler.transform(X.flatten().reshape(-1, 1)).reshape(X.shape) for X in X_test])
    return X_train_scaled, X_test_scaled


def divide_input_data(data):
    try:
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.15, random_state=4280)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Divide input data error: {e}")


def perform_find_peaks(X_array):
    result_X_test = [scipy.signal.find_peaks(sublist)[0] for sublist in X_array]
    middle_index = min(len(sublist) for sublist in result_X_test) // 2
    elements1 = [[sublist[i] for i in range(max(0, middle_index - 50), middle_index)] for sublist in result_X_test]
    elements2 = [[sublist[i] for i in range(middle_index, min(len(sublist), middle_index + 50))] for sublist in
                 result_X_test]
    return [sublist1 + sublist2 for sublist1, sublist2 in zip(elements1, elements2)]


def data_setup():
    try:
        data = load_chunked_data_npy()
        (X_train, X_test, y_train, y_test) = divide_input_data(data)
        (X_train_scaled, X_test_scaled) = scaling_Xs_wavelength_reflectance(X_train, X_test)

        # y_train = np.round(y_train, 3)
        # y_test = np.round(y_test, 3)

        y_test_rounded = []
        for sublist in y_test:
            rounded_first = np.round(sublist[0], 3)
            rounded_third = np.round(sublist[2], 3)
            y_test_rounded.append(np.array([rounded_first, sublist[1], rounded_third, sublist[3]]))
        y_test_rounded = np.reshape(np.array(y_test_rounded), (-1, 1, 4, 15))

        y_train_rounded = []
        for sublist in y_train:
            rounded_first = np.round(sublist[0], 3)
            rounded_third = np.round(sublist[2], 3)
            y_train_rounded.append(np.array([rounded_first, sublist[1], rounded_third, sublist[3]]))
        y_train_rounded = np.reshape(np.array(y_train_rounded), (-1, 1, 4, 15))

        # SECTION: dla bilstm + tylko reflektancje
        # X_train_reshaped = X_train_scaled.reshape(-1, 1, 300)
        # X_test_reshaped = X_test_scaled.reshape(-1, 1, 300)
        return X_test, X_train_scaled, X_test_scaled, y_train_rounded, y_test_rounded
        # return X_test, X_train_scaled, X_test_scaled, y_train, y_test
    except Exception:
        print("Data setup error")
