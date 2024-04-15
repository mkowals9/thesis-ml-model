from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


class Metrics:
    def __init__(self):
        self.epochs = 0

        self.root_mean_squared_error = []
        self.val_root_mean_squared_error = []
        self.root_mean_squared_error_cal = 0

        self.mean_absolute_error_cal = 0
        self.mean_absolute_error = []
        self.val_mean_absolute_error = []

        self.mean_squared_error = []
        self.val_mean_squared_error = []
        self.mean_squared_error_cal = 0

        self.loss = []
        self.val_loss = []

        self.training_config = {}

        self.logcosh = []
        self.val_logcosh = []

        self.mean_absolute_percentage_error = []
        self.val_mean_absolute_percentage_error = []

        self.mean_squared_logarithmic_error = []
        self.val_mean_squared_logarithmic_error = []

    def calculate(self, y_test=None, y_pred=None):
        try:
            # y_pred_matched = np.squeeze(y_pred, axis=1)
            # y_test_matched = np.squeeze(y_pred, axis=1)
            mse_cal = mean_squared_error(y_test, y_pred)
            print("mse: ", mse_cal)
            self.mean_squared_error_cal = mse_cal
            rmse_cal = np.sqrt(mse_cal)
            print("rmse: ", rmse_cal)
            self.root_mean_squared_error_cal = rmse_cal
            mae_cal = mean_absolute_error(y_test, y_pred)
            print("mae: ", mae_cal)
            self.mean_absolute_error_cal = mae_cal
        except Exception as e:
            print(f"An error has occurred during calculations: {e}")

    def save_history_training_data(self, config, history=None):
        try:
            self.epochs = len(history.history['loss'])
            # self.logcosh = history.history["logcosh"]
            self.loss = history.history["loss"]
            self.mean_absolute_error = history.history["mean_absolute_error"]
            # self.mean_absolute_percentage_error = history.history["mean_absolute_percentage_error"]
            self.mean_squared_error = history.history["mean_squared_error"]
            # self.mean_squared_logarithmic_error = history.history["mean_squared_logarithmic_error"]
            self.root_mean_squared_error = history.history["root_mean_squared_error"]
            # self.val_logcosh = history.history["val_logcosh"]
            self.val_loss = history.history["val_loss"]
            self.val_mean_absolute_error = history.history["val_mean_absolute_error"]
            # self.val_mean_absolute_percentage_error = history.history["val_mean_absolute_percentage_error"]
            self.val_mean_squared_error = history.history["val_mean_squared_error"]
            # self.val_mean_squared_logarithmic_error = history.history["val_mean_squared_logarithmic_error"]
            self.val_root_mean_squared_error = history.history["val_root_mean_squared_error"]
            self.training_config = config
        except Exception as e:
            print(f"History training data wasn't saved due to error: {e}")
