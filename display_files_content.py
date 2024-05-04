import json
import matplotlib.pyplot as plt
import datetime
import random
import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

FILE_KERAS_MODEL_PATH = "/home/marcelina/Desktop/uniform_cnn_1714761378_93348/cnn_nn_model_trained_model_1714761378_93348.keras"

FILE_TRAINING_PATH = '/home/marcelina/Desktop/uniform_cnn_1714761378_93348/model_training_output_1714761378_93348_cnn_nn_model.json'

FILE_DATA_JSON_PATH = ('/home/marcelina/Documents/misc/master-thesis/new_code/stats/1708903520_261761'
                  '/model_output_1708903520_261761_base_neural_network_4_outputs.json')

FILE_DATA_Y_TEST_PATH = '/home/marcelina/Desktop/uniform_cnn_1714761378_93348/model_output_1714761378_93348_cnn_nn_model_y_test.py.npy'
FILE_DATA_Y_PRED_PATH = '/home/marcelina/Desktop/uniform_cnn_1714761378_93348/model_output_1714761378_93348_cnn_nn_model_y_predicted.py.npy'


def display_plot():
    with open(FILE_TRAINING_PATH, 'r') as json_file:
        training_data = json.load(json_file)

    epochs_range = range(1, training_data["epochs"] + 1)

    metric_name = "mae"
    val_metric_name = "val_" + metric_name
    metric_name_formatted = metric_name.capitalize() if metric_name == "loss" else metric_name.upper()
    save_plots = False
    ct = datetime.datetime.now().timestamp()
    ct = str(ct).replace(".", "_")

    plt.plot(epochs_range, training_data[metric_name], label=f'{metric_name_formatted} - zbiór treningowy')
    plt.plot(epochs_range, training_data[val_metric_name], label=f'{metric_name_formatted} - zbiór walidacyjny')
    # plt.yscale('log')
    plt.xlabel('Epoki')
    plt.ylabel(metric_name_formatted)
    plt.title(f'{metric_name_formatted} - zbiór treningowy i walidacyjny')
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig(f'./plots/{metric_name}_{ct}.png')
        plt.clf()
    else:
        plt.show()
        plt.clf()


def load_json_data():
    with open(FILE_DATA_JSON_PATH, 'r') as json_file:
        data = json.load(json_file)
    print("loaded data from json")
    return data


def load_npy_data():
    np_new_load = lambda *a, **k: np.load(*a, allow_pickle=True, **k)
    y_test = np_new_load(FILE_DATA_Y_TEST_PATH)
    y_pred = np_new_load(FILE_DATA_Y_PRED_PATH)
    return {"y_test": y_test, "y_predicted": y_pred}


def display_predictions():
    # data = load_json_data()
    data = load_npy_data()
    indices = list(range(len(data["y_test"])))
    random_indexes = random.sample(indices, 5)
    y_test_random = []
    y_pred_random = []
    for ind in random_indexes:
        y_test_random.append(data["y_test"][ind].tolist())
        y_pred_random.append(data["y_predicted"][ind].tolist())
    print("chosen random values")
    mae = []
    mse = []
    rmse = []
    for indd in list(range(len(y_test_random))):
        mae.append(mean_absolute_error(y_test_random[indd], y_pred_random[indd]))
        msee = mean_squared_error(y_test_random[indd], y_pred_random[indd])
        mse.append(msee)
        rmse.append(np.sqrt(msee))
    loaded_model = keras.saving.load_model(FILE_KERAS_MODEL_PATH)
    print(" - - - Keras model - - - ")
    loaded_model.summary()
    to_save = {"y_test": y_test_random, "y_pred": y_pred_random, "mae": mae, "mse": mse, "rmse": rmse}
    with open("./outputs.json", "w") as outfile:
        json.dump(to_save, outfile, indent=4)


if __name__ == "__main__":
    display_plot()
    display_predictions()
