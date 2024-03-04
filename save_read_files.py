import json
import ijson
from msgspec.json import decode
import msgspec
import os
import numpy as np

DATA_MODEL_NON_UNIFORM_1_MLN = '/home/marcelina/Documents/misc/model_inputs/non_uniform/npys'
TRAINING_CONFIG_JSON = './training_config.json'


class Measurement(msgspec.Struct):
    reflectance: list[float]
    wavelengths: list[float]
    n_eff: list[float]
    delta_n_eff: list[float]
    X_z: list[float]
    period: list[float]


def load_data_from_jsons():
    try:
        np_new_load = lambda *a, **k: np.load(*a, allow_pickle=True, **k)
        filenames = [name for name in os.listdir(DATA_MODEL_NON_UNIFORM_1_MLN) if name.find('.npy')]
        first_file_path = os.path.join(DATA_MODEL_NON_UNIFORM_1_MLN, filenames[0])
        data = np_new_load(first_file_path)
        # for filename in filenames[1:]:
        filename = filenames[1]
        file_path = os.path.join(DATA_MODEL_NON_UNIFORM_1_MLN, filename)
        array = np_new_load(file_path)
        data = np.concatenate((data, array), axis=0)
        del array
        return data
    except Exception as e:
        print(e)
        print("Data setup error")


def load_training_config():
    with open(TRAINING_CONFIG_JSON, 'r') as json_file:
        training_config = json.load(json_file)
    return training_config


def save_all_to_files(model_metrics, X_test, y_test, y_predicted, ct, nn_trained):
    try:
        output_training = {
            "epochs": model_metrics.epochs,

            # "rmse": model_metrics.root_mean_squared_error,
            # "val_rmse": model_metrics.val_root_mean_squared_error,
            # "rmse_cal": model_metrics.root_mean_squared_error_cal,

            "mse": model_metrics.mean_squared_error,
            "val_mse": model_metrics.val_mean_squared_error,
            "mse_cal": model_metrics.mean_squared_error_cal,

            "mae": model_metrics.mean_absolute_error,
            "val_mae": model_metrics.val_mean_absolute_error,
            "mae_cal": model_metrics.mean_absolute_error_cal,

            "loss": model_metrics.loss,
            "val_loss": model_metrics.val_loss,

            # "logcosh": model_metrics.logcosh,
            # "val_logcosh": model_metrics.val_logcosh,

            #"mean_absolute_percentage_error": model_metrics.mean_absolute_percentage_error,
            #"val_mean_absolute_percentage_error": model_metrics.val_mean_absolute_percentage_error,

            # "mean_squared_logarithmic_error": model_metrics.mean_squared_logarithmic_error,
            # "val_mean_squared_logarithmic_error": model_metrics.val_mean_squared_logarithmic_error,

            "config": model_metrics.training_config,
            "note": "[x,y] zamiast y, danych wszystkich bylo 400k"
        }

        output_results = {
            "X_test": X_test.tolist(),
            "y_test": y_test.tolist(),
            "y_predicted": y_predicted.tolist()
        }

        with open(f"./trainings/{ct}/model_training_output_{ct}_{nn_trained.model_name}.json", "w") as outfile:
            json.dump(output_training, outfile, indent=4)

        with open(f"./trainings/{ct}/model_output_{ct}_{nn_trained.model_name}.json", "w") as outfile:
            json.dump(output_results, outfile, indent=4)

        filename_model = f"{nn_trained.model_name}_trained_model_" + ct + ".keras"
        nn_trained.model.save(f"./trainings/{ct}/{filename_model}")
        print(f"All metrics, data and model saved successfully in {ct} folder and in {filename_model}!")
    except Exception:
        print("Save files error")


def convert_json_to_npy():
    filenames = ['data_model_input_last.json']
    for filename in filenames:
        file_path = os.path.join(DATA_MODEL_NON_UNIFORM_1_MLN, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        array_data = np.array(list(data.values()))
        np.save(DATA_MODEL_NON_UNIFORM_1_MLN + '/data_model_input_last.npy', array_data)


# if __name__ == "__main__":
#     convert_json_to_npy()
