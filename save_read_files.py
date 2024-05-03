import json
import re
import os
import numpy as np

from plots import make_new_directory

# NONUNIFORM
DATA_MODEL_NON_UNIFORM_1_MLN = '/home/marcelina/Documents/misc/model_inputs/non_uniform/npys'
DATA_MODEL_1ST_CHUNKED = '/home/marcelina/Documents/misc/model_inputs/pierwsze_chunked_data'
DATA_MODEL_GAUSS = '/home/marcelina/Documents/misc/model_inputs/gauss'
DATA_MODEL_CORRECT_GAUSS = '/home/marcelina/Documents/misc/model_inputs/gauss_correct'
DATA_MODEL_GAUSS_MORE_RANDOM = '/home/marcelina/Documents/misc/model_inputs/gauss_more_random'
DATA_MODEL_FIRST_PARABOLIC = '/home/marcelina/Documents/misc/model_inputs/parabolic_first'
DATA_MODEL_PARABOLIC_SCALED = '/home/marcelina/Documents/misc/model_inputs/parabol_different_scales'
DATA_MODEL_POLYNOMIAL_MIXED = '/home/marcelina/Documents/misc/model_inputs/polynomial_2_3_first_mixed'
DATA_MODEL_COEFFICIENTS = '/home/marcelina/Documents/misc/model_inputs/new_coef'
DATA_MODEL_POLYNOMIAL_MIXED_TOTAL_RANDOM = '/home/marcelina/Documents/misc/model_inputs/polynomial_2_3_all_random_coef'
DATA_MODEL_NO_UNIQUE = '/home/marcelina/Documents/misc/model_inputs/2_3_degree_polynomial_without_unique'
DATA_MODEL_SINUSOID = '/home/marcelina/Documents/misc/model_inputs/sinusoid_first'
DATA_MODEL_SINUSOID_MORE = '/home/marcelina/Documents/misc/model_inputs/sinusoid_more_shifts'


# UNIFORM
DATA_MODEL_INPUT_JSON = '/home/marcelina/Documents/misc/model_inputs/data_model_input.json'
DATA_MODEL_40_PARAM_INPUT_WITH_X_Z_ = '/home/marcelina/Documents/misc/model_inputs/data_model_40_param_input_with_X_z_'
DATA_MODEL_25_PARAM_INPUT_WITH_X_Z_ = '/home/marcelina/Documents/misc/model_inputs/data_model_25_param_input_with_X_z_'

TRAINING_CONFIG_JSON = './training_config.json'


def convert_to_decibels(data):
    return 10 * np.log10(data)


def extract_chunk_index(filename):
    match = re.search(r'chunk_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None


def parameters_in_chunk_to_list(chunk_dict, folder_name):
    np_new_load = lambda *a, **k: np.load(*a, allow_pickle=True, **k)
    X_z = np_new_load(os.path.join(folder_name, chunk_dict["X_z"][0]))
    delta_n_eff = np_new_load(os.path.join(folder_name, chunk_dict["delta_n_eff"][0]))
    n_eff = np_new_load(os.path.join(folder_name, chunk_dict["n_eff"][0]))
    period = np_new_load(os.path.join(folder_name, chunk_dict["period"][0]))
    reflectances = np_new_load(os.path.join(folder_name, chunk_dict["reflectances"][0]))
    # reflectances = convert_to_decibels(bare_ref)
    # wavelengths = np_new_load(os.path.join(DATA_MODEL_FIRST_PARABOLIC, chunk_dict["wavelengths"][0]))
    return reflectances, n_eff, period, X_z, delta_n_eff


def coefficients_in_chunk_to_list(chunk_dict, folder_name):
    np_new_load = lambda *a, **k: np.load(*a, allow_pickle=True, **k)
    coefficients = np_new_load(os.path.join(folder_name, chunk_dict["coefficients"][0]))
    reflectances = np_new_load(os.path.join(folder_name, chunk_dict["reflectances"][0]))
    # wavelengths = np_new_load(os.path.join(DATA_MODEL_FIRST_PARABOLIC, chunk_dict["wavelengths"][0]))
    return reflectances, coefficients


def load_chunked_data_npy(param_name: str):
    # TODO !!!! REMEMBER TO CHANGE FOLDER NAME !!!!!
    if param_name == "coefficients":
        folder_name = DATA_MODEL_COEFFICIENTS
    else:
        folder_name = DATA_MODEL_SINUSOID_MORE
    # CODE IF WE WANT TO HAVE AS THE OUTPUT A, B, C, D
    filenames = os.listdir(folder_name)
    chunks = {}

    for filename in filenames:
        chunk_index = extract_chunk_index(filename)
        if chunk_index is not None:
            data_type = re.search(r'model_input_(\w+)_chunk', filename).group(1)
            if chunk_index not in chunks:
                chunks[chunk_index] = {}
            chunks[chunk_index].setdefault(data_type, []).append(filename)
    chunks_list = list(chunks.values())

    if param_name == "coefficients":
        loaded_chunk_data = [coefficients_in_chunk_to_list(chunk_dict, folder_name) for chunk_dict in chunks_list]
    else:
        loaded_chunk_data = [parameters_in_chunk_to_list(chunk_dict, folder_name) for chunk_dict in chunks_list]

    # folder_name = DATA_MODEL_POLYNOMIAL_MIXED
    # filenames = os.listdir(folder_name)
    # chunks = {}
    #
    # for filename in filenames:
    #     chunk_index = extract_chunk_index(filename)
    #     if chunk_index is not None:
    #         data_type = re.search(r'model_input_(\w+)_chunk', filename).group(1)
    #         if chunk_index not in chunks:
    #             chunks[chunk_index] = {}
    #         chunks[chunk_index].setdefault(data_type, []).append(filename)
    # chunks_list = list(chunks.values())
    #
    # if param_name == "coefficients":
    #     loaded_chunk_data_2 = [coefficients_in_chunk_to_list(chunk_dict, folder_name) for chunk_dict in chunks_list]
    # else:
    #     loaded_chunk_data_2 = [parameters_in_chunk_to_list(chunk_dict, folder_name) for chunk_dict in chunks_list]
    #
    # loaded_chunk_data.extend(loaded_chunk_data_2)

    # folder_name = DATA_MODEL_PARABOLIC_SCALED
    # filenames = os.listdir(folder_name)
    # chunks = {}
    #
    # for filename in filenames:
    #     chunk_index = extract_chunk_index(filename)
    #     if chunk_index is not None:
    #         data_type = re.search(r'model_input_(\w+)_chunk', filename).group(1)
    #         if chunk_index not in chunks:
    #             chunks[chunk_index] = {}
    #         chunks[chunk_index].setdefault(data_type, []).append(filename)
    # chunks_list = list(chunks.values())
    # loaded_chunk_data = [parameters_in_chunk_to_list(chunk_dict, folder_name) for chunk_dict in chunks_list]

    # ONLY 1/4 OF ALL DATA
    # w = 4
    # subarray_length = len(loaded_chunk_data) // w
    # start_index = len(loaded_chunk_data) // 2
    #
    # loaded_chunk_data = loaded_chunk_data[start_index:start_index + subarray_length]

    # tylko reflektancje w osi X
    X_data = np.array([sublist for object_data in loaded_chunk_data for sublist in object_data[0]])

    # (X,Y) w X_data
    # reflectances = np.array([val for object_data in loaded_chunk_data for val in object_data[3]])
    # wavelengths = loaded_chunk_data[0][2][0]
    # X_data = np.empty((len(reflectances), len(reflectances[0]), 2))
    # for i, reflectance in enumerate(reflectances):
    #     X_data[i, :, 0] = wavelengths
    #     X_data[i, :, 1] = reflectance

    # EVERYTHING IN Y_DATA
    if param_name == "n_eff":
        n_effs = np.array([val for object_data in loaded_chunk_data for val in object_data[1]])
        y_data = n_effs
    elif param_name == "period":
        periods = np.array([val for object_data in loaded_chunk_data for val in object_data[2]])
        y_data = periods
    elif param_name == "X_z":
        Xzs = np.array([val for object_data in loaded_chunk_data for val in object_data[3]])
        y_data = Xzs
    elif param_name == "delta_n_eff":
        delta_n_effs = np.array([val for object_data in loaded_chunk_data for val in object_data[4]])
        y_data = delta_n_effs
    elif param_name == "coefficients":
        coefficients = np.array([np.round(val, 4) for object_data in loaded_chunk_data for val in object_data[1]])
        y_data = coefficients
    else:
        n_effs = np.array([val for object_data in loaded_chunk_data for val in object_data[1]])
        periods = np.array([val for object_data in loaded_chunk_data for val in object_data[2]])
        Xzs = np.array([val for object_data in loaded_chunk_data for val in object_data[3]])
        delta_n_effs = np.array([val for object_data in loaded_chunk_data for val in object_data[4]])
        y_data = np.array([np.concatenate((
            value,
            periods[i],
            Xzs[i],
            delta_n_effs[i]
        )) for i, value in enumerate(n_effs)])
    return X_data, y_data


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

            # "mean_absolute_percentage_error": model_metrics.mean_absolute_percentage_error,
            # "val_mean_absolute_percentage_error": model_metrics.val_mean_absolute_percentage_error,

            # "mean_squared_logarithmic_error": model_metrics.mean_squared_logarithmic_error,
            # "val_mean_squared_logarithmic_error": model_metrics.val_mean_squared_logarithmic_error,

            "config": model_metrics.training_config,
            "note": f"{nn_trained.model_name} siec, przeskalowane, n_eff, no k-fold, sinusoidy z y shift"
        }

        output_results = {
            "X_test": X_test.tolist(),
            "y_test": y_test.tolist(),
            "y_predicted": y_predicted.tolist()
        }
        make_new_directory(ct)

        with open(f"./trainings/{ct}/model_training_output_{ct}_{nn_trained.model_name}.json", "w") as outfile:
            json.dump(output_training, outfile, indent=4)

        with open(f"./trainings/{ct}/model_output_{ct}_{nn_trained.model_name}.json", "w") as outfile:
            json.dump(output_results, outfile, indent=4)

        filename_model = f"{nn_trained.model_name}_trained_model_" + ct + ".keras"
        nn_trained.model.save(f"./trainings/{ct}/{filename_model}")

        np.save(f"./trainings/{ct}/model_output_{ct}_{nn_trained.model_name}_X_test.py",
                np.array(X_test.tolist()))
        np.save(f"./trainings/{ct}/model_output_{ct}_{nn_trained.model_name}_y_test.py",
                np.array(y_test.tolist()))
        np.save(f"./trainings/{ct}/model_output_{ct}_{nn_trained.model_name}_y_predicted.py",
                np.array(y_predicted.tolist()))
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


def load_uniform_gratings_jsons():
    data = []
    try:
        with open('%s0.json' % DATA_MODEL_25_PARAM_INPUT_WITH_X_Z_, 'r') as json_file_1:
            data.extend(json.load(json_file_1))
        with open('%s1.json' % DATA_MODEL_25_PARAM_INPUT_WITH_X_Z_, 'r') as json_file_2:
            data.extend(json.load(json_file_2))
        with open('%s2.json' % DATA_MODEL_25_PARAM_INPUT_WITH_X_Z_, 'r') as json_file_3:
            data.extend(json.load(json_file_3))
        with open('%s3.json' % DATA_MODEL_25_PARAM_INPUT_WITH_X_Z_, 'r') as json_file_4:
            data.extend(json.load(json_file_4))
        with open('%s4.json' % DATA_MODEL_25_PARAM_INPUT_WITH_X_Z_, 'r') as json_file_5:
            data.extend(json.load(json_file_5))
        with open('%s5.json' % DATA_MODEL_25_PARAM_INPUT_WITH_X_Z_, 'r') as json_file_6:
            data.extend(json.load(json_file_6))
    except Exception as e:
        print(f"Error while reading jsons with uniform gratings: {str(e)}")
    return data


# if __name__ == "__main__":
#     convert_json_to_npy()
