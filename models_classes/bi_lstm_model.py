import keras
from keras.models import Sequential
from tensorflow.keras import optimizers
from keras.layers import Dense, LSTM, Bidirectional


class BiLstmModel:

    def create_standard_model(self):
        model = Sequential([
            Bidirectional(LSTM(500, activation='relu', input_shape=self.input_shape, return_sequences=True)),
            Bidirectional(LSTM(300, activation='relu', return_sequences=True)),
            Bidirectional(LSTM(300, activation='relu', return_sequences=False)),
            Dense(200, activation='relu'),
            Dense(200, activation='linear')
        ])

        root_mean_squared_error = keras.metrics.RootMeanSquaredError()
        mean_squared_error = keras.metrics.MeanSquaredError()
        mean_absolute_error = keras.metrics.MeanAbsoluteError()
        mean_absolute_percentage_error = keras.metrics.MeanAbsolutePercentageError()
        mean_squared_logarithmic_error = keras.metrics.MeanSquaredLogarithmicError()
        log_cosh_error = keras.metrics.LogCoshError()
        model.compile(optimizer=optimizers.Adam(lr=0.001, decay=1e-6),
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
