import matplotlib.pyplot as plt
import os


def make_new_directory(ct):
    try:
        new_directory = f"./trainings/{ct}"
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
    except Exception:
        print("New trainings directory hasn't been created")


def save_training_stats_as_plots_in_files(epochs_range, metrics, ct, save_plots=True):
    try:
        make_new_directory(ct)

        # Plot training and validation loss
        plt.plot(epochs_range, metrics.loss, label='Training Loss')
        plt.plot(epochs_range, metrics.val_loss, label='Validation Loss')
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

        # Plot training and validation mae
        plt.plot(epochs_range, metrics.root_mean_squared_error, label='Training RMSE')
        plt.plot(epochs_range, metrics.val_root_mean_squared_error, label='Validation RMSE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title('Training and Validation Root MSE')
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig(f'./trainings/{ct}/root_mse_{ct}.png')
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
