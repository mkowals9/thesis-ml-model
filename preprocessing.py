import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import scipy
import random

from save_read_files import load_chunked_data_npy, load_uniform_gratings_jsons

TRAINING_CONFIG_JSON = './training_config.json'

sc = StandardScaler()


def env_setup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ["XDG_SESSION_TYPE"] = "xcb"

    # tf.config.threading.set_inter_op_parallelism_threads(16)


def generate_800_wavelenghts():
    num_points = 800
    start_value = 1.45e-6 * 1e6  # początkowy zakres fal
    end_value = 1.6e-6 * 1e6  # końcowy zakres fal
    return np.linspace(start_value, end_value, num_points)


def convert_to_decibels(data):
    return np.array([math.log10(x) * 10 for x in data])


def scaling_Xs_only_reflectance(X_train, X_test):
    sc.fit(X_train)
    return sc.transform(X_train), sc.transform(X_test)


def scaling_Xs_wavelength_reflectance(X_train, X_test):
    scaler = StandardScaler()

    X_train_concatenated = np.concatenate([X.flatten() for X in X_train]).reshape(-1, 1)
    scaler.fit(X_train_concatenated)

    X_train_scaled = np.array([scaler.transform(X.flatten().reshape(-1, 1)).reshape(X.shape) for X in X_train])
    X_test_scaled = np.array([scaler.transform(X.flatten().reshape(-1, 1)).reshape(X.shape) for X in X_test])
    return X_train_scaled, X_test_scaled


def scaling_Xs(X_train, X_test):
    return (np.array([sc.fit_transform(X) for X in X_train]),
            np.array([sc.fit_transform(X) for X in X_test]))


def divide_input_data(data):
    try:
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1],
                                                            test_size=0.15, random_state=11235813)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Divide input data error: {e}")


def divide_input_data_uniform_case(data):
    try:
        random.shuffle(data)
        wavelengths = generate_800_wavelenghts()
        max_wavelength = max(wavelengths)
        X_data = []
        for data_object in data:
            X_temp = []
            for index, ref in enumerate(data_object["reflectance"]):
                X_temp.append([wavelengths[index] / max_wavelength, ref])
            X_data.append(X_temp)
        y_data = [[data_object["n_eff"], data_object["delta_n_eff"], data_object["period"], data_object["X_z"]]
                  for data_object in data]

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=4280)

        y_train = [[round(sublist[0], 3), sublist[1] * 1e3,
                    sublist[2] * 1e6, round(sublist[3], 2)] for sublist in y_train]
        y_test = [[round(sublist[0], 3), sublist[1] * 1e3,
                   sublist[2] * 1e6, round(sublist[3], 2)] for sublist in y_test]

        y_train = [[sublist[0], round(sublist[1], 3),
                    round(sublist[2], 3), sublist[3]] for sublist in y_train]
        y_test = [[sublist[0], round(sublist[1], 3),
                   round(sublist[2], 3), sublist[3]] for sublist in y_test]

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        return X_train, X_test, y_train, y_test
    except Exception:
        print("Divide input data error")


def perform_find_peaks(X_array):
    result_X_test = [scipy.signal.find_peaks(sublist)[0] for sublist in X_array]
    middle_index = min(len(sublist) for sublist in result_X_test) // 2
    elements1 = [[sublist[i] for i in range(max(0, middle_index - 50), middle_index)] for sublist in result_X_test]
    elements2 = [[sublist[i] for i in range(middle_index, min(len(sublist), middle_index + 50))] for sublist in
                 result_X_test]
    return [sublist1 + sublist2 for sublist1, sublist2 in zip(elements1, elements2)]


def data_setup_nonuniform(param_name: str):
    try:
        data = load_chunked_data_npy(param_name)
        (X_train, X_test, y_train, y_test) = divide_input_data(data)
        (X_train_scaled, X_test_scaled) = scaling_Xs_wavelength_reflectance(X_train, X_test)

        # y_train = np.round(y_train, 3)
        # y_test = np.round(y_test, 3)

        # y_test_rounded = []
        # for sublist in y_test:
        #     rounded_first = np.round(sublist[0], 3)
        #     rounded_third = np.round(sublist[2], 3)
        #     y_test_rounded.append(np.array([rounded_first, sublist[1], rounded_third, sublist[3]]))
        # y_test_rounded = np.reshape(np.array(y_test_rounded), (-1, 1, 4, 15))
        #
        # y_train_rounded = []
        # for sublist in y_train:
        #     rounded_first = np.round(sublist[0], 3)
        #     rounded_third = np.round(sublist[2], 3)
        #     y_train_rounded.append(np.array([rounded_first, sublist[1], rounded_third, sublist[3]]))
        # y_train_rounded = np.reshape(np.array(y_train_rounded), (-1, 1, 4, 15))

        #y_test_rounded = np.reshape(np.array(y_test), (-1, 1, 15))
        #y_train_rounded = np.reshape(np.array(y_train), (-1, 1, 15))

        # SECTION: only bilstm + only reflectances
        # X_train_reshaped = X_train_scaled.reshape(-1, 1, 300)
        # X_test_reshaped = X_test_scaled.reshape(-1, 1, 300)
        return X_test, X_train_scaled, X_test_scaled, y_train, y_test
        # return X_test, X_train_scaled, X_test_scaled, y_train, y_test
    except Exception as e:
        print(f"Data setup error : {e}")


def data_setup_uniform():
    # 800 elementów jako długości fal
    try:
        data = load_uniform_gratings_jsons()

        (X_train, X_test, y_train, y_test) = divide_input_data_uniform_case(data)
        (X_train_scaled, X_test_scaled) = scaling_Xs(X_train, X_test)

        X_train_reshaped = X_train_scaled.reshape(len(X_train_scaled), -1)
        X_test_reshaped = X_test_scaled.reshape(len(X_test_scaled), -1)
        return X_test, X_train_reshaped, X_test_reshaped, y_train, y_test
    except Exception as e:
        print(f"Data setup error: {str(e)}")
