import matplotlib.pyplot as plt
import os
import random
import numpy as np


def make_new_directory(ct):
    try:
        new_directory = f"./trainings/{ct}"
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
    except Exception as e:
        print(f"New trainings directory hasn't been created: {e}")


def save_training_stats_as_plots_in_files(epochs_range, metrics, ct, save_plots=False):
    try:
        make_new_directory(ct)

        # Plot training and validation loss
        plt.plot(epochs_range, metrics.loss, label='Loss - zbiór treningowy')
        plt.plot(epochs_range, metrics.val_loss, label='Loss - zbiór walidacyjny')
        # plt.yscale('log')
        plt.xlabel('Epoki')
        plt.ylabel('Loss')
        plt.title('Loss - zbiór treningowy i walidacyjny')
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig(f'./trainings/{ct}/loss_{ct}.png')
            plt.clf()
        else:
            plt.show()
            plt.clf()

        # Plot training and validation mse
        plt.plot(epochs_range, metrics.mean_squared_error, label='MSE - zbiór treningowy')
        plt.plot(epochs_range, metrics.val_mean_squared_error, label='MSE - zbiór walidacyjny')
        # plt.yscale('log')
        plt.xlabel('Epoki')
        plt.ylabel('MSE')
        plt.title('MSE - zbiór treningowy i walidacyjny')
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig(f'./trainings/{ct}/mse_{ct}.png')
            plt.clf()
        else:
            plt.show()
            plt.clf()

        # Plot training and validation mae
        plt.plot(epochs_range, metrics.mean_absolute_error, label='MAE - zbiór treningowy')
        plt.plot(epochs_range, metrics.val_mean_absolute_error, label='MAE - zbiór walidacyjny')
        # plt.yscale('log')
        plt.xlabel('Epoki')
        plt.ylabel('MAE')
        plt.title('MAE - zbiór treningowy i walidacyjny')
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig(f'./trainings/{ct}/mae_{ct}.png')
            plt.clf()
        else:
            plt.show()
            plt.clf()

        # Plot training and validation rmse
        plt.plot(epochs_range, metrics.root_mean_squared_error, label='RMSE - zbiór treningowy')
        plt.plot(epochs_range, metrics.val_root_mean_squared_error, label='RMSE - zbiór walidacyjny')
        # plt.yscale('log')
        plt.xlabel('Epoki')
        plt.ylabel('RMSE')
        plt.title('RMSE - zbiór treningowy i walidacyjny')
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig(f'./trainings/{ct}/rmse_{ct}.png')
            plt.clf()
        else:
            plt.show()
            plt.clf()

        print("All metrics plots saved successfully!")
    except Exception as e:
        print(f"An error has occurred during metrics plots creation: {e}")


# only one parameter on the output, the length = number of sections
def plot_predicted_actual_single_array_values(y_predicted, y_actual, ct, param_name, save_plots=False):
    try:
        random_values = random.sample(range(0, len(y_predicted)), 20)
        sections = np.arange(1, len(y_predicted[0]) + 1) if y_predicted.shape[1] == 15 else np.arange(1, 16)
        for example_index in random_values:
            plt.plot(sections, y_predicted[example_index], drawstyle='steps-post', label='Przewidziane wartości')
            plt.plot(sections, y_actual[example_index], drawstyle='steps-post', label='Rzeczywiste wartości')
            plt.xlabel('Indeksy sekcji')
            plt.ylabel(f'{param_name}')
            plt.title(f'Przewidziane i rzeczywiste wartości - parameter {param_name}')
            plt.legend()
            plt.grid(True)
            if save_plots:
                plt.savefig(f'./trainings/{ct}/predicted_vs_actual_{param_name}_{example_index}_{ct}.png')
                plt.clf()
            else:
                plt.show()
                plt.clf()
    except Exception as e:
        print(f"Param plots {param_name} from single array error: {e}")


# output array has length 60, so 4 parameters with 15 values
def separate_predicted_actual_values_from_one_array_and_plot(y_predicted, y_actual, ct, save_plots=False,
                                                             param_name="n_eff"):
    try:
        # y_pred_matched = np.squeeze(y_predicted, axis=1)
        # y_test_matched = np.squeeze(y_actual, axis=1)
        random_values = random.sample(range(0, len(y_predicted)), 20)
        sections = np.arange(1, len(y_predicted[0]) + 1) if y_predicted.shape[1] == 15 else np.arange(1, 16)
        for example_index in random_values:
            # param_name = "n_eff"
            param_index = 0
            plot_actual_predicted_for_param_from_one_big_param_array(ct, example_index, save_plots, sections,
                                                                     y_actual, y_predicted, param_index, param_name)
            param_name = "period"
            param_index = 1
            plot_actual_predicted_for_param_from_one_big_param_array(ct, example_index, save_plots, sections,
                                                                     y_actual, y_predicted, param_index, param_name)

            param_name = "X_z"
            param_index = 2
            plot_actual_predicted_for_param_from_one_big_param_array(ct, example_index, save_plots, sections,
                                                                     y_actual, y_predicted, param_index, param_name)

            param_name = "delta_n_eff"
            param_index = 3
            plot_actual_predicted_for_param_from_one_big_param_array(ct, example_index, save_plots, sections,
                                                                     y_actual, y_predicted, param_index, param_name)
        print("All predictions plots saved successfully")
    except Exception as e:
        print(f"Param plots from one big array error: {e}")


# output array of length 60, so 4 parameters with 15 values
def plot_actual_predicted_for_param_from_one_big_param_array(ct, example_index, save_plots, sections, y_actual,
                                                             y_pred_matched, param_index, param_name):
    try:
        if param_index == 0 and param_name == "n_eff":
            temp_y_pred = y_pred_matched[example_index][0:15]
            temp_y_actual = y_actual[example_index][0:15]
        elif param_index == 1 and param_name == "period":
            temp_y_pred = y_pred_matched[example_index][15:30]
            temp_y_actual = y_actual[example_index][15:30]
        elif param_index == 2 and param_name == "X_z":
            temp_y_pred = y_pred_matched[example_index][30:45]
            temp_y_actual = y_actual[example_index][30:45]
        elif param_index == 3 and param_name == "delta_n_eff":
            temp_y_pred = y_pred_matched[example_index][45:60]
            temp_y_actual = y_actual[example_index][45:60]
        else:
            temp_y_pred = y_pred_matched
            temp_y_actual = y_actual
        plt.plot(sections, temp_y_pred, drawstyle='steps-post', label='Predicted values')
        plt.plot(sections, temp_y_actual, drawstyle='steps-post', label='Actual values')
        plt.xlabel('Section')
        plt.ylabel(f'{param_name}')
        plt.title(f'Predicted and actual values - parameter {param_name}')
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig(f'./trainings/{ct}/predicted_vs_actual_{param_name}_{example_index}_{ct}.png')
            plt.clf()
        else:
            plt.show()
            plt.clf()
        print("Successfully saved actual_predicted plots for params")
    except Exception as e:
        print(f"Failed to save actual_predicted plots from subarrays: {e}")


def generate_polynomial_y_from_coef_and_scale(coeffs, x_values, desired_min, desired_max):
    values = np.polyval(coeffs, x_values)
    min_value = np.min(values)
    max_value = np.max(values)
    value_ = (max_value - min_value)
    if value_ != 0:
        normalized_values = [(value - min_value) / (max_value - min_value) * (desired_max - desired_min) + desired_min
                             for value in values]
        normalized_values = [desired_min if value < desired_min else value for value in normalized_values]
        normalized_values = [desired_max if value > desired_max else value for value in normalized_values]
        return normalized_values


def plot_from_coefficients(y_predicted, y_test, ct, save_plots):
    try:
        random_values = random.sample(range(0, len(y_predicted)-1), 20)
        sections = np.arange(1, len(y_predicted[0]) + 1) if y_predicted.shape[1] == 15 else np.arange(1, 16)
        L = 4e-3 * 1e3  # change if the L changes in data generation
        x_values = np.linspace(-L / 2, L / 2, 15)
        for example_index in random_values:
            y_pred = y_predicted[example_index]
            y_actual = y_test[example_index]
            for i in range(0, 4):
                if i == 0:
                    pred_values = generate_polynomial_y_from_coef_and_scale(y_pred[0:4], x_values, 1.44, 1.45)
                    actual_values = generate_polynomial_y_from_coef_and_scale(y_actual[0:4], x_values, 1.44, 1.45)
                    param_name = "n_eff"
                elif i == 1:
                    pred_values = generate_polynomial_y_from_coef_and_scale(y_pred[4:8], x_values, 0.1, 1)
                    actual_values = generate_polynomial_y_from_coef_and_scale(y_actual[4:8], x_values, 0.1, 1)
                    param_name = "delta_n_eff"
                elif i == 2:
                    pred_values = generate_polynomial_y_from_coef_and_scale(y_pred[8:12], x_values, 0.1, 9.9)
                    actual_values = generate_polynomial_y_from_coef_and_scale(y_actual[8:12], x_values, 0.1, 9.9)
                    param_name = "X_z"
                elif i == 3:
                    pred_values = generate_polynomial_y_from_coef_and_scale(y_pred[12:16], x_values, 5.350, 5.400)
                    actual_values = generate_polynomial_y_from_coef_and_scale(y_actual[12:16], x_values, 5.350, 5.400)
                    param_name = "period"
                plt.plot(sections, pred_values, marker='o', drawstyle='steps-post', label='Przewidziane wartości')
                plt.plot(sections, actual_values, marker='o', drawstyle='steps-post', label='Rzeczywiste wartości')
                plt.xlabel('Indeksy sekcji')
                plt.ylabel('Wartości')
                plt.legend()
                plt.title(f'Przewidziane i rzeczywiste wartości parameteru {param_name}')
                plt.grid(True)
                if save_plots:
                    plt.savefig(
                        f'./trainings/{ct}/predicted_vs_actual_coefficient_{example_index}_{param_name}_{ct}.png')
                    plt.clf()
                else:
                    plt.show()
                    plt.clf()
        print("All predictions plots saved successfully")
    except Exception as e:
        print(f"Failed to plot from coefficients: {e}")
