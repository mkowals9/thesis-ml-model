from preprocessing import env_setup, data_setup
from save_read_files import load_training_config

env_setup()
training_config = load_training_config()
X_test, X_train_reshaped, X_test_reshaped, y_train, y_test = data_setup()