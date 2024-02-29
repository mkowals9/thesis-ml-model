import json

DATA_MODEL_25_PARAM_INPUT_WITH_X_Z_ = '/home/marcelina/Documents/misc/model_inputs/data_model_25_param_input_with_X_z_'
TRAINING_CONFIG_JSON = './training_config.json'


def load_data_from_jsons():
    try:
        data = []
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
        return data
    except Exception:
        print("Data setup error")


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
            "rmse_cal": model_metrics.root_mean_squared_error_cal,

            "mse": model_metrics.mean_squared_error,
            "val_mse": model_metrics.val_mean_squared_error,
            "mse_cal": model_metrics.mean_squared_error_cal,

            "mae": model_metrics.mean_absolute_error,
            "val_mae": model_metrics.val_mean_absolute_error,
            "mae_cal": model_metrics.mean_absolute_error_cal,

            "loss": model_metrics.loss,
            "val_loss": model_metrics.val_loss,

            "logcosh": model_metrics.logcosh,
            "val_logcosh": model_metrics.val_logcosh,

            "mean_absolute_percentage_error": model_metrics.mean_absolute_percentage_error,
            "val_mean_absolute_percentage_error": model_metrics.val_mean_absolute_percentage_error,

            "mean_squared_logarithmic_error": model_metrics.mean_squared_logarithmic_error,
            "val_mean_squared_logarithmic_error": model_metrics.val_mean_squared_logarithmic_error,

            "config": model_metrics.training_config,
            "note": ""
        }

        output_results = {
            "X_test": X_test.tolist(),
            "y_test": y_test.tolist(),
            "y_predicted": y_predicted.tolist()
        }

        with open(f"./trainings/{ct}/model_training_output_{ct}_{nn_trained.model_name}.json", "w") as outfile:
            json.dump(output_training, outfile, indent=4)

        with open(f"./stats/{ct}/model_output_{ct}_{nn_trained.model_name}.json", "w") as outfile:
            json.dump(output_results, outfile, indent=4)

        filename_model = f"{nn_trained.model_name}_trained_model_" + ct + ".keras"
        nn_trained.model.save(f"./trainings/{ct}/{filename_model}")
        print(f"All metrics, data and model saved successfully in {ct} folder and in {filename_model}!")
    except Exception:
        print("Save files error")
