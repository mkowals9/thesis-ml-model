import matplotlib.pyplot as plt
import os
import random


def make_new_directory(ct):
    try:
        new_directory = f"./trainings/{ct}"
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
    except Exception:
        print("New trainings directory hasn't been created")


def save_training_stats_as_plots_in_files(epochs_range, metrics, ct, save_plots=False):
    try:
        make_new_directory(ct)

        # Plot training and validation loss
        plt.plot(epochs_range, metrics.loss, label='Training Loss')
        plt.plot(epochs_range, metrics.val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig(f'./trainings/{ct}/loss_{ct}.png')
            plt.clf()
        else:
            plt.show()
            plt.clf()

        # Plot training and validation mse
        plt.plot(epochs_range, metrics.mean_squared_error, label='Training MSE')
        plt.plot(epochs_range, metrics.val_mean_squared_error, label='Validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('Training and Validation MSE')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig(f'./trainings/{ct}/mse_{ct}.png')
            plt.clf()
        else:
            plt.show()
            plt.clf()

        # Plot training and validation mae
        plt.plot(epochs_range, metrics.mean_absolute_error, label='Training MAE')
        plt.plot(epochs_range, metrics.val_mean_absolute_error, label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title('Training and Validation MAE')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig(f'./trainings/{ct}/mae_{ct}.png')
            plt.clf()
        else:
            plt.show()
            plt.clf()

        print("All plots saved successfully!")
    except Exception:
        print("An error has occurred during plots creation")


def plot_predicted_actual_values(sections, y_predicted, y_actual, ct, save_plots=False):
    random_values = random.sample(range(0, len(y_predicted)), 10)
    for example_index in random_values:
        for i in range(0, 4):
            if i == 0:
                param_name = "n_eff"
            elif i == 1:
                param_name = "delta_n_eff"
            elif i == 2:
                param_name = "period"
            else:
                param_name = "X_z"
            plt.plot(sections, y_predicted[example_index][i], drawstyle='steps-post', label='Predicted values')
            plt.plot(sections, y_actual[example_index][i], drawstyle='steps-post', label='Actual values')
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
