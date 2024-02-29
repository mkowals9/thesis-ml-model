import keras
import tensorflow
from keras import Input
from keras.models import Sequential
from keras.src.layers import Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.python.keras import regularizers

class BiLstmModel:

    def create_standard_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            Conv1D(filters=40, kernel_size=2, activation='relu'), #64
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=2, activation='relu'), #40
            MaxPooling1D(pool_size=2),
            Conv1D(filters=25, kernel_size=2, activation='relu'), #32
            MaxPooling1D(pool_size=2),
            Conv1D(filters=20, kernel_size=2, activation='relu'), #20
            MaxPooling1D(pool_size=2),
            Conv1D(filters=16, kernel_size=2, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.1),
            Flatten(),
            Dense(self.output_dim, activation='linear')
        ])

        root_mean_squared_error = keras.metrics.RootMeanSquaredError()
        mean_squared_error = keras.metrics.MeanSquaredError()
        mean_absolute_error = keras.metrics.MeanAbsoluteError()
        mean_absolute_percentage_error = keras.metrics.MeanAbsolutePercentageError()
        mean_squared_logarithmic_error = keras.metrics.MeanSquaredLogarithmicError()
        log_cosh_error = keras.metrics.LogCoshError()
        model.compile(optimizer='adam',
                      metrics=[root_mean_squared_error, mean_squared_error,
                               mean_absolute_error, mean_absolute_percentage_error,
                               mean_squared_logarithmic_error, log_cosh_error],
                      loss='mean_squared_error')
        model.summary()

        self.model = model


    def __init__(self):
        # shape of initial is n,800,1 [y1, y2, .. ]
        # input_shape = (800, 1)
        # shape of initial is n,800,2 [[x1, y1], .. ]
        self.input_shape = (1600, 1)
        self.output_dim = 4
        self.model = None
        self.create_standard_model()
        self.model_name = "bi_lstm_model"