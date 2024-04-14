import datetime

from sklearn.model_selection import KFold
from keras.src.callbacks import EarlyStopping
from metrics import Metrics
from models_classes.basic_dense_model import BasicDenseModel
from models_classes.lstm_model import LstmModel
from models_classes.cnn_model import CnnModel
from plots import save_training_stats_as_plots_in_files, plot_predicted_actual_single_array_values, \
    separate_predicted_actual_values_from_one_array_and_plot, plot_from_coefficients
from preprocessing import env_setup, data_setup
from save_read_files import load_training_config, save_all_to_files
import numpy as np


def run_training_with_callbacks_and_k_folds():
    neural_network = BasicDenseModel()

    X_test, X_test_reshaped, X_train_reshaped, training_config, y_test, y_train = prepare_data()

    early_stopping_loss = EarlyStopping(monitor='loss', patience=5, verbose=1, mode='auto')
    early_stopping_val_loss = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    callbacks = [early_stopping_loss, early_stopping_val_loss]

    print("~ ~ Training start ~ ~")
    kf = KFold(n_splits=training_config["k_folds"])
    mse_scores = []
    histories = []
    models = []
    for train_index, val_index in kf.split(X_train_reshaped):
        X_train_fold, X_val_fold = X_train_reshaped[train_index], X_train_reshaped[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        print(f"~ ~ Training start fold {train_index} ~ ~")
        neural_network = BasicDenseModel()
        history = neural_network.model.fit(X_train_fold, y_train_fold,
                                           batch_size=training_config["batch_size"],
                                           epochs=training_config["epochs"],
                                           validation_split=0.05,
                                           verbose=1,
                                           callbacks=callbacks)
        models.append(neural_network)
        mse = neural_network.model.evaluate(X_val_fold, y_val_fold, verbose=0)[0]
        mse_scores.append(mse)
        histories.append(history)

    avg_mse = np.mean(mse_scores)
    print("Average MSE:", avg_mse)

    ct = datetime.datetime.now().timestamp()
    ct = str(ct).replace(".", "_")

    best_model_index = np.argmin(mse_scores)
    print("Best model index and its mse:", best_model_index, mse_scores[best_model_index])

    # with open(f"./trainings/{ct}/model_training_output_histories.json", "w") as outfile:
    #    json.dump(histories, outfile, indent=4)

    # perform_after_training_actions(X_test, X_test_reshaped, X_train_fold, history, training_config, y_test, y_train)


def run_training_with_callbacks(param_name: str, model_index: int):
    if model_index == 1:
        neural_network = BasicDenseModel()
    elif model_index == 2:
        neural_network = CnnModel()
    elif model_index == 3:
        neural_network = LstmModel()
    X_test, X_test_reshaped, X_train_reshaped, training_config, y_test, y_train = prepare_data(param_name)

    early_stopping_loss = EarlyStopping(monitor='loss', patience=5, verbose=1, mode='auto')
    early_stopping_val_loss = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
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
                                   param_name, neural_network)


def run_training_without_callbacks(param_name: str, model_index: int):
    if model_index == 1:
        neural_network = BasicDenseModel()
    elif model_index == 2:
        neural_network = CnnModel()
    elif model_index == 3:
        neural_network = LstmModel()
    X_test, X_test_reshaped, X_train_reshaped, training_config, y_test, y_train = prepare_data(param_name)

    print("~ ~ Training start ~ ~")

    history = neural_network.model.fit(X_train_reshaped, y_train,
                                       batch_size=training_config["batch_size"],
                                       epochs=training_config["epochs"],
                                       validation_split=0.09,
                                       verbose=1,
                                       )

    perform_after_training_actions(X_test, X_test_reshaped, X_train_reshaped, history, training_config, y_test, y_train,
                                   param_name, neural_network)


def perform_after_training_actions(X_test, X_test_reshaped, X_train_reshaped, history, training_config, y_test,
                                   y_train, param_name, neural_network):
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

        print("~ ~ Plots ~ ~")
        epochs_range = range(1, model_metrics.epochs + 1)
        ct = datetime.datetime.now().timestamp()
        ct = str(ct).replace(".", "_")
        save_training_stats_as_plots_in_files(epochs_range, model_metrics, ct, training_config["save_plots"])

        # when we have coefficients a, b, c, d
        plot_from_coefficients(y_predicted, y_test, ct, training_config["save_plots"])

        # when we have n_eff, delta_n_eff, etc. in one
        # plot_predicted_actual_many_arrays_values(y_predicted, y_test, ct, training_config["save_plots"], param_name)

        # when we have only one parameter as the output, e.g. n_eff lub delta_n_eff
        # plot_predicted_actual_single_array_values(y_predicted, y_test, ct, param_name, training_config["save_plots"])

        print("~ ~ Saving to files predictions and models ~ ~")
        save_all_to_files(model_metrics, X_test, y_test, y_predicted, ct, neural_network)
    except Exception as e:
        print(f"Metrics calculation error: {e}")


def prepare_data(param_name: str):
    training_config = load_training_config()
    X_test, X_train_reshaped, X_test_reshaped, y_train, y_test = data_setup(param_name)
    print(f"Size of train dataset: {len(X_train_reshaped)}")
    print(f"Size of test dataset: {len(X_test)}")
    return X_test, X_test_reshaped, X_train_reshaped, training_config, y_test, y_train


if __name__ == "__main__":
    env_setup()
    # run_training_without_callbacks("coefficients", 2)
    # run_training_with_callbacks("coefficients", 2)
    # run_training_without_callbacks("coefficients", 2)
    run_training_with_callbacks("coefficients", 1)
    run_training_with_callbacks("coefficients", 2)
    run_training_with_callbacks("coefficients", 3)
    # run_training_without_callbacks("coefficients", 1)

    # run_training_without_callbacks("n_eff", 1)
    # run_training_without_callbacks("n_eff", 2)
    # run_training_without_callbacks("X_z", 1)
    # run_training_without_callbacks("X_z", 2)

    # run_training_without_callbacks("delta_n_eff", 1)
    # run_training_without_callbacks("delta_n_eff", 2)

    # run_training_without_callbacks("period", 1)
    # run_training_without_callbacks("period", 2)

    # run_training_with_callbacks("period", 2)
    # run_training_with_callbacks("Xz", 2)
    # run_training_with_callbacks("n_eff", 2)
