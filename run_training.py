import datetime
from sklearn.model_selection import KFold
from keras.src.callbacks import EarlyStopping
from metrics import Metrics
from model_classes_nonuniform_gratings.basic_dense_model import BasicDenseModel
from model_classes_nonuniform_gratings.gru_model import GruModel
from model_classes_nonuniform_gratings.lstm_model import LstmModel
from model_classes_nonuniform_gratings.cnn_model import CnnModel
from model_classes_uniform_gratings.base_neural_network import BaseNeuralNetwork
from model_classes_uniform_gratings.cnn_neural_network import CnnNeuralNetwork
from plots import save_training_stats_as_plots_in_files, plot_predicted_actual_single_array_values, \
    separate_predicted_actual_values_from_one_array_and_plot, plot_from_coefficients
from preprocessing import env_setup, data_setup_nonuniform, data_setup_uniform
from save_read_files import load_training_config, save_all_to_files
import numpy as np


def choose_nn_model_for_nonuniform(model_index: int):
    if model_index == 1:
        return BasicDenseModel()
    elif model_index == 2:
        return CnnModel()
    elif model_index == 3:
        return LstmModel()
    elif model_index == 4:
        return GruModel()
    else:
        return None


def choose_nn_model_for_uniform(model_index: int):
    if model_index == 1:
        return BaseNeuralNetwork()
    elif model_index == 2:
        return CnnNeuralNetwork()


def run_training_with_callbacks_and_with_k_folds(param_name: str, model_index: int, is_uniform: bool):
    X_test, X_test_reshaped, X_train_reshaped, training_config, y_test, y_train = prepare_data(param_name, is_uniform)

    early_stopping_loss = EarlyStopping(monitor='loss', patience=5, verbose=1, mode='auto')
    early_stopping_val_loss = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    callbacks = [early_stopping_loss, early_stopping_val_loss]

    print("~ ~ Training start ~ ~")
    kf = KFold(n_splits=training_config["k_folds"])
    loss_metrics_scores = []
    histories = []
    models = []
    for train_index, val_index in kf.split(X_train_reshaped):
        X_train_fold, X_val_fold = X_train_reshaped[train_index], X_train_reshaped[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        print(f"~ ~ Training start fold {train_index} ~ ~")
        neural_network = choose_nn_model_for_nonuniform(model_index) if not is_uniform \
            else choose_nn_model_for_uniform(model_index)
        print(f"Size training fold dataset: {len(X_train_fold)}")
        print(f"Size validation fold dataset: {len(X_val_fold)}")
        history = neural_network.model.fit(X_train_fold, y_train_fold,
                                           batch_size=training_config["batch_size"],
                                           epochs=training_config["epochs"],
                                           validation_split=0.05,
                                           verbose=1,
                                           callbacks=callbacks)
        models.append(neural_network)
        # from evaluate I have [loss, mean_absolute_error, mean_squared_error, root_mean_squared_error]
        loss_metric = neural_network.model.evaluate(X_val_fold, y_val_fold, verbose=0)[0]
        loss_metrics_scores.append(loss_metric)
        histories.append(history)

    avg_loss_metric = np.mean(loss_metrics_scores)
    print("Average loss metric:", avg_loss_metric)

    best_model_index = np.argmin(loss_metrics_scores)
    print("Best model index and its loss metric:", best_model_index, loss_metrics_scores[best_model_index])

    # ct = datetime.datetime.now().timestamp()
    # ct = str(ct).replace(".", "_")
    # with open(f"./trainings/{ct}/model_training_output_histories.json", "w") as outfile:
    #    json.dump(histories, outfile, indent=4)

    perform_after_training_actions(X_test, X_test_reshaped, X_train_reshaped, histories[best_model_index],
                                   training_config, y_test, y_train, param_name, models[best_model_index], is_uniform)


def run_training_without_callbacks_and_with_k_folds(param_name: str, model_index: int, is_uniform: bool):
    X_test, X_test_reshaped, X_train_reshaped, training_config, y_test, y_train = prepare_data(param_name, is_uniform)

    print("~ ~ Training start ~ ~")
    kf = KFold(n_splits=training_config["k_folds"])
    loss_metrics_scores = []
    histories = []
    models = []
    for train_index, val_index in kf.split(X_train_reshaped):
        X_train_fold, X_val_fold = X_train_reshaped[train_index], X_train_reshaped[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        print(f"~ ~ Training start fold {train_index} ~ ~")
        neural_network = choose_nn_model_for_nonuniform(model_index) if not is_uniform \
            else choose_nn_model_for_uniform(model_index)
        print(f"Size training fold dataset: {len(X_train_fold)}")
        print(f"Size validation fold dataset: {len(X_val_fold)}")
        history = neural_network.model.fit(X_train_fold, y_train_fold,
                                           batch_size=training_config["batch_size"],
                                           epochs=training_config["epochs"],
                                           validation_split=0.05,
                                           verbose=1)
        models.append(neural_network)
        # from evaluate I have [loss, mean_absolute_error, mean_squared_error, root_mean_squared_error]
        loss_metric = neural_network.model.evaluate(X_val_fold, y_val_fold, verbose=0)[0]
        loss_metrics_scores.append(loss_metric)
        histories.append(history)

    avg_loss_metric = np.mean(loss_metrics_scores)
    print("Average loss metric:", avg_loss_metric)

    best_model_index = np.argmin(loss_metrics_scores)
    print("Best model index and its loss metric:", best_model_index, loss_metrics_scores[best_model_index])

    # ct = datetime.datetime.now().timestamp()
    # ct = str(ct).replace(".", "_")
    # with open(f"./trainings/{ct}/model_training_output_histories.json", "w") as outfile:
    #    json.dump(histories, outfile, indent=4)

    perform_after_training_actions(X_test, X_test_reshaped, X_train_reshaped, histories[best_model_index],
                                   training_config, y_test, y_train, param_name, models[best_model_index], is_uniform)


def run_training_with_callbacks(param_name: str, model_index: int, is_uniform: bool):
    neural_network = choose_nn_model_for_nonuniform(model_index) if not is_uniform \
        else choose_nn_model_for_uniform(model_index)
    X_test, X_test_reshaped, X_train_reshaped, training_config, y_test, y_train = prepare_data(param_name, is_uniform)

    early_stopping_loss = EarlyStopping(monitor='loss', patience=15, verbose=1, mode='min')
    early_stopping_val_loss = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    callbacks = [early_stopping_loss, early_stopping_val_loss]

    print("~ ~ Training start ~ ~")

    history = neural_network.model.fit(X_train_reshaped, y_train,
                                       batch_size=training_config["batch_size"],
                                       epochs=training_config["epochs"],
                                       validation_split=0.05,
                                       verbose=1,
                                       callbacks=callbacks
                                       )

    perform_after_training_actions(X_test, X_test_reshaped, X_train_reshaped, history, training_config, y_test, y_train,
                                   param_name, neural_network, is_uniform)


def run_training_without_callbacks(param_name: str, model_index: int, is_uniform: bool):
    neural_network = choose_nn_model_for_nonuniform(model_index) if not is_uniform \
        else choose_nn_model_for_uniform(model_index)
    X_test, X_test_reshaped, X_train_reshaped, training_config, y_test, y_train = prepare_data(param_name, is_uniform)

    print("~ ~ Training start ~ ~")
    history = neural_network.model.fit(X_train_reshaped, y_train,
                                       batch_size=training_config["batch_size"],
                                       epochs=training_config["epochs"],
                                       validation_split=0.09,
                                       verbose=1,
                                       )

    perform_after_training_actions(X_test, X_test_reshaped, X_train_reshaped, history, training_config, y_test, y_train,
                                   param_name, neural_network, is_uniform)


def perform_after_training_actions(X_test, X_test_reshaped, X_train_reshaped, history, training_config, y_test,
                                   y_train, param_name, neural_network, is_uniform):
    try:
        # print("~ ~ Metrics calculation start ~ ~")
        # score = neural_network.model.evaluate(X_train_reshaped, y_train, verbose=1)
        # print("Loss training:", score[1])
        # score = neural_network.model.evaluate(X_test_reshaped, y_test, verbose=1)
        # print("Loss test:", score[1])
        print("~ ~ Predictions and saving history data ~ ~")
        y_predicted = neural_network.model.predict(X_test_reshaped, training_config["batch_size"])
        model_metrics = Metrics()
        model_metrics.calculate(y_test, y_predicted)
        model_metrics.save_history_training_data(training_config, history)
        ct = datetime.datetime.now().timestamp()
        ct = str(ct).replace(".", "_")

        print("~ ~ Saving to files predictions and models ~ ~")
        save_all_to_files(model_metrics, X_test, y_test, y_predicted, ct, neural_network)

        print("~ ~ Plots ~ ~")
        epochs_range = range(1, model_metrics.epochs + 1)
        save_training_stats_as_plots_in_files(epochs_range, model_metrics, ct, training_config["save_plots"])
        if not is_uniform:
            # when we have coefficients a, b, c, d
            if param_name == "coefficients":
                plot_from_coefficients(y_predicted, y_test, ct, training_config["save_plots"])

            # when we have n_eff, delta_n_eff, etc. in one
            elif param_name == "all":
                separate_predicted_actual_values_from_one_array_and_plot(y_predicted, y_test, ct,
                                                                         training_config["save_plots"], param_name)
            else:
                # when we have only one parameter as the output, e.g. n_eff lub delta_n_eff
                plot_predicted_actual_single_array_values(y_predicted, y_test, ct, param_name,
                                                          training_config["save_plots"])

    except Exception as e:
        print(f"Metrics calculation error: {e}")


def prepare_data(param_name: str, is_uniform: bool):
    training_config = load_training_config()
    X_test, X_train_reshaped, X_test_reshaped, y_train, y_test = data_setup_nonuniform(param_name) if not is_uniform \
        else data_setup_uniform()
    print(f"Size of train dataset: {len(X_train_reshaped)}")
    print(f"Size of test dataset: {len(X_test)}")
    return X_test, X_test_reshaped, X_train_reshaped, training_config, y_test, y_train


if __name__ == "__main__":
    env_setup()
    # run_training_without_callbacks("n_eff", 4, False)
    # run_training_without_callbacks_and_with_k_folds("n_eff", 2, False)

    # TODO !!!! REMEMBER TO CHANGE FOLDER NAME IF YOU CHANGE PARAM_NAME (FILE SAVE_READ_FILES) !!!!
    # TODO !!!! AND OUTPUT DIMENSIONS OF MODELS !!!!

    # run_training_without_callbacks("coefficients", 1, False)
    # run_training_with_callbacks("coefficients", 2)
    # run_training_without_callbacks("coefficients", 2)
    # run_training_with_callbacks("coefficients", 1)
    # run_training_with_callbacks("coefficients", 2)
    # run_training_with_callbacks("n_eff", 4, False)
    # run_training_without_callbacks("period", 1, False)

    # TODO do pracki
    run_training_without_callbacks("n_eff", 4, False)
    # run_training_without_callbacks("delta_n_eff", 2, False)
    # run_training_without_callbacks("period", 2, False)
    # run_training_without_callbacks("X_z", 2, False)
    # run_training_without_callbacks("coefficients", 2, False)




