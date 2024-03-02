import datetime

from keras.src.callbacks import EarlyStopping

from metrics import Metrics
from models_classes.bi_lstm_model import BiLstmModel
from plots import save_training_stats_as_plots_in_files
from preprocessing import env_setup, data_setup
from save_read_files import load_training_config, save_all_to_files

env_setup()
training_config = load_training_config()
X_test, X_train_reshaped, X_test_reshaped, y_train, y_test = data_setup()

early_stopping_loss = EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min')
early_stopping_mean_absolute_percentage_error = EarlyStopping(monitor='mean_absolute_percentage_error', patience=8,
                                                              verbose=1, mode='auto')
early_stopping_loss_rmse = EarlyStopping(monitor='root_mean_squared_error', patience=5, verbose=1, mode='max')
callbacks = [early_stopping_loss, early_stopping_mean_absolute_percentage_error]

print("~ ~ Training start ~ ~")
print(f"Size of train dataset: {len(X_train_reshaped)}")
print(f"Size of test dataset: {len(X_test)}")
neural_network = BiLstmModel()
history = neural_network.model.fit(X_train_reshaped, y_train,
                                   batch_size=training_config["batch_size"],
                                   epochs=training_config["epochs"],
                                   validation_split=0.1,
                                   verbose=1,
                                   callbacks=callbacks)

print("~ ~ Metrics calculation start ~ ~")
score = neural_network.model.evaluate(X_train_reshaped, y_train, verbose=1)
print("Loss training:", score[1])
score = neural_network.model.evaluate(X_test_reshaped, y_test, verbose=1)
print("Loss test:", score[1])

y_predicted = neural_network.model.predict(X_test_reshaped, training_config["batch_size"])

model_metrics = Metrics()
model_metrics.calculate(y_test, y_predicted)
model_metrics.save_history_training_data(training_config, history)

print("~ ~ Plots ~ ~")
epochs_range = range(1, model_metrics.epochs + 1)
ct = datetime.datetime.now().timestamp()
ct = str(ct).replace(".", "_")

save_training_stats_as_plots_in_files(epochs_range, model_metrics, ct, training_config["save_plots"])

print("~ ~ Saving to files start ~ ~")
save_all_to_files(model_metrics, X_test, y_test, y_predicted, ct, neural_network)