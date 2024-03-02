import keras
from keras import Input
from keras.models import Sequential
from keras.src.layers import Reshape
from tensorflow import optimizers
from keras.layers import Dense, LSTM, Bidirectional


class BiLstmModel:

    def create_standard_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            Bidirectional(LSTM(500, activation='relu', return_sequences=True)),
            Bidirectional(LSTM(300, activation='relu', return_sequences=True)),
            Bidirectional(LSTM(300, activation='relu', return_sequences=False)),
            Dense(200, activation='relu'),
            Dense(200, activation='linear'),
            Dense(4 * 50, activation='linear'),  # Output layer with a shape of 4 lists of 50 elements
            Reshape((4, 50))  # Reshape the output to get four lists of 50 elements
        ])

        root_mean_squared_error = keras.metrics.RootMeanSquaredError()
        mean_squared_error = keras.metrics.MeanSquaredError()
        mean_absolute_error = keras.metrics.MeanAbsoluteError()
        mean_absolute_percentage_error = keras.metrics.MeanAbsolutePercentageError()
        mean_squared_logarithmic_error = keras.metrics.MeanSquaredLogarithmicError()
        log_cosh_error = keras.metrics.LogCoshError()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      metrics=[mean_squared_error, mean_absolute_error, mean_absolute_percentage_error],
                      loss='mean_squared_error')
        model.summary()

        self.model = model

    def __init__(self):
        # shape of initial is n,800,1 [y1, y2, .. ]
        # input_shape = (800, 1)
        # shape of initial is n,800,2 [[x1, y1], .. ]
        # self.input_shape = (1600, 1)
        self.input_shape = (1, 500,)
        # self.output_dim = 4
        self.model = None
        self.create_standard_model()
        self.model_name = "bi_lstm_model"
