import json
import re
import os
import numpy as np
import math

DATA_MODEL_NON_UNIFORM_1_MLN = '/home/marcelina/Documents/misc/model_inputs/non_uniform/npys'
DATA_MODEL_1ST_CHUNKED = '/home/marcelina/Documents/misc/model_inputs/pierwsze_chunked_data'
DATA_MODEL_GAUSS = '/home/marcelina/Documents/misc/model_inputs/gauss'
DATA_MODEL_CORRECT_GAUSS = '/home/marcelina/Documents/misc/model_inputs/gauss_correct'
DATA_MODEL_GAUSS_MORE_RANDOM = '/home/marcelina/Documents/misc/model_inputs/gauss_more_random'

TRAINING_CONFIG_JSON = './training_config.json'


def convert_to_decibels(data):
    return 10 * np.log10(data)


def extract_chunk_index(filename):
    match = re.search(r'chunk_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None


def chunk_to_list(chunk_dict):
    np_new_load = lambda *a, **k: np.load(*a, allow_pickle=True, **k)
    X_z = np_new_load(os.path.join(DATA_MODEL_GAUSS_MORE_RANDOM, chunk_dict["X_z"][0]))
    delta_n_eff = np_new_load(os.path.join(DATA_MODEL_GAUSS_MORE_RANDOM, chunk_dict["delta_n_eff"][0]))
    n_eff = np_new_load(os.path.join(DATA_MODEL_GAUSS_MORE_RANDOM, chunk_dict["n_eff"][0]))
    period = np_new_load(os.path.join(DATA_MODEL_GAUSS_MORE_RANDOM, chunk_dict["period"][0]))
    reflectances = np_new_load(os.path.join(DATA_MODEL_GAUSS_MORE_RANDOM, chunk_dict["reflectances"][0]))
    #reflectances = convert_to_decibels(bare_ref)
    wavelengths = np_new_load(os.path.join(DATA_MODEL_GAUSS_MORE_RANDOM, chunk_dict["wavelengths"][0]))
    return n_eff, period, wavelengths, reflectances, X_z, delta_n_eff


def load_chunked_data_npy():
    filenames = os.listdir(DATA_MODEL_GAUSS_MORE_RANDOM)
    chunks = {}

    for filename in filenames:
        chunk_index = extract_chunk_index(filename)
        if chunk_index is not None:
            data_type = re.search(r'model_input_(\w+)_chunk', filename).group(1)
            if chunk_index not in chunks:
                chunks[chunk_index] = {}
            chunks[chunk_index].setdefault(data_type, []).append(filename)
    chunks_list = list(chunks.values())

    loaded_chunk_data_org = [chunk_to_list(chunk_dict) for chunk_dict in chunks_list]
    # tylko 1/2 danych
    subarray_length = len(loaded_chunk_data_org) // 2
    start_index = len(loaded_chunk_data_org) // 2  # Choosing the middle as the starting index, you can choose any other valid index as well

    loaded_chunk_data = loaded_chunk_data_org[start_index:start_index + subarray_length]

    #tylko reflektancje w osi X
    #X_data = np.array([sublist for object_data in loaded_chunk_data for sublist in object_data[3]])

    # (X,Y) w X_data
    reflectances = np.array([val for object_data in loaded_chunk_data for val in object_data[3]])
    wavelengths = loaded_chunk_data_org[0][2][0] * 1e9
    X_data = np.empty((len(reflectances), len(reflectances[0]), 2))
    for i, reflectance in enumerate(reflectances):
        X_data[i, :, 0] = wavelengths  # Assign wavelengths to the first column
        X_data[i, :, 1] = reflectance

    # wszystko w y_data
    n_effs = np.array([val for object_data in loaded_chunk_data for val in object_data[0]])
    periods = np.array([val for object_data in loaded_chunk_data for val in object_data[1]])
    Xzs = np.array([val for object_data in loaded_chunk_data for val in object_data[4]])
    delta_n_effs = np.array([val for object_data in loaded_chunk_data for val in object_data[5]])
    y_data = np.array(
        [np.array([n_effs[i], periods[i], Xzs[i], delta_n_effs[i]]) for i in range(len(n_effs))])
    return X_data, y_data


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
        print(f"Data setup error: {e}")


def load_training_config():
    with open(TRAINING_CONFIG_JSON, 'r') as json_file:
        training_config = json.load(json_file)
    return training_config


def save_all_to_files(model_metrics, X_test, y_test, y_predicted, ct, nn_trained):
    try:
        output_training = {
            "epochs": model_metrics.epochs,

            "rmse": model_metrics.root_mean_squared_error,
            "val_rmse": model_metrics.val_root_mean_squared_error,
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
            "note": "bilstm, gauss"
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
    except Exception as e:
        print(f"Save files error: {e}")


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
