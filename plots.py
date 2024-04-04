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
        plt.plot(epochs_range, metrics.loss, label='Training Loss')
        plt.plot(epochs_range, metrics.val_loss, label='Validation Loss')
        # plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig(f'./trainings/{ct}/loss_{ct}.png')
            plt.clf()
        else:
            plt.show()
            plt.clf()

        # Plot training and validation mse
        # plt.plot(epochs_range, metrics.mean_squared_error, label='Training MSE')
        # plt.plot(epochs_range, metrics.val_mean_squared_error, label='Validation MSE')
        # plt.yscale('log')
        # plt.xlabel('Epochs')
        # plt.ylabel('MSE')
        # plt.title('Training and Validation MSE')
        # plt.legend()
        # plt.grid(True)
        # if save_plots:
        #     plt.savefig(f'./trainings/{ct}/mse_{ct}.png')
        #     plt.clf()
        # else:
        #     plt.show()
        #     plt.clf()

        # Plot training and validation mae
        plt.plot(epochs_range, metrics.mean_absolute_error, label='Training MAE')
        plt.plot(epochs_range, metrics.val_mean_absolute_error, label='Validation MAE')
        # plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title('Training and Validation MAE')
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig(f'./trainings/{ct}/mae_{ct}.png')
            plt.clf()
        else:
            plt.show()
            plt.clf()

        # Plot training and validation rmse
        plt.plot(epochs_range, metrics.root_mean_squared_error, label='Training RMSE')
        plt.plot(epochs_range, metrics.val_root_mean_squared_error, label='Validation RMSE')
        # plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.title('Training and Validation RMSE')
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
        print(f"An error has occurred during plots creation: {e}")


def plot_predicted_actual_single_array_values(y_predicted, y_actual, ct, save_plots=False):
    try:
        random_values = random.sample(range(0, len(y_predicted)), 10)
        sections = np.arange(1, len(y_predicted[0][0]) + 1)
        y_pred_matched = np.squeeze(y_predicted, axis=1)
        y_test_matched = np.squeeze(y_actual, axis=1)
        for example_index in random_values:
            param_name = "n_eff"
            plt.plot(sections, y_pred_matched[example_index], drawstyle='steps-post', label='Predicted values')
            plt.plot(sections, y_test_matched[example_index], drawstyle='steps-post', label='Actual values')
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
    except Exception as e:
        print(f"Param plots error: {e}")


def plot_predicted_actual_many_arrays_values(y_predicted, y_actual, ct, save_plots=False):
    try:
        #y_pred_matched = np.squeeze(y_predicted, axis=1)
        #y_test_matched = np.squeeze(y_actual, axis=1)
        random_values = random.sample(range(0, len(y_predicted)), 10)
        sections = np.arange(1, len(y_predicted[0]) + 1)
        for example_index in random_values:
            param_name = "delta_n_eff"
            param_index = 0
            plot_actual_predicted_for_param(ct, example_index, save_plots, sections,
                                            y_actual[0:15], y_predicted[0:15], param_name)
            # param_name = "period"
            # param_index = 1
            # plot_actual_predicted_for_param(ct, example_index, save_plots, sections,
            #                                 y_actual[15:30], y_predicted[15:30], param_index, param_name)
            #
            # param_name = "X_z"
            # param_index = 2
            # plot_actual_predicted_for_param(ct, example_index, save_plots, sections,
            #                                 y_actual[30:45], y_predicted[30:45], param_index, param_name)
            #
            # param_name = "delta_n_eff"
            # param_index = 3
            # plot_actual_predicted_for_param(ct, example_index, save_plots, sections,
            #                                 y_actual[45:60], y_predicted[45:60], param_index, param_name)

    except Exception as e:
        print(f"Param plots error: {e}")


def plot_actual_predicted_for_param(ct, example_index, save_plots, sections, y_actual, y_pred_matched,
                                    param_name):
    plt.plot(sections, y_pred_matched[example_index], drawstyle='steps-post', label='Predicted values')
    plt.plot(sections, y_actual[example_index], drawstyle='steps-post', label='Actual values')
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
