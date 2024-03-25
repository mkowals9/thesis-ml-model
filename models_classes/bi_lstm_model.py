import keras
from keras import Input
from keras.models import Sequential
from keras.src.layers import Reshape, Dropout
from keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras import regularizers


class BiLstmModel:

    def create_standard_model(self):
        l2_lambda = 0.2
        output_size = (4, 15)

        model = Sequential([
            Input(shape=self.input_shape),
            Bidirectional(LSTM(300, activation='relu', return_sequences=True)),
            Dropout(rate=0.5),
            Bidirectional(LSTM(100, activation='relu', return_sequences=True)),
            Dropout(rate=0.5),
            # Bidirectional(LSTM(200, activation='relu', return_sequences=False)),
            Dense(100, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
            # Dense(15, activation='relu'),
            # jesli na wyjsciu mamy jeden array
            # Dense(15, activation='linear', kernel_regularizer=regularizers.l2(l2_lambda)),
            Dense(4 * 15, activation='relu'),
            Reshape((-1,) + output_size)
        ])

        mean_squared_error = keras.metrics.MeanSquaredError()
        root_mean_squared_error = keras.metrics.RootMeanSquaredError()
        mean_absolute_error = keras.metrics.MeanAbsoluteError()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-7),
                      metrics=[mean_absolute_error, root_mean_squared_error],
                      loss='mean_squared_error')
        model.summary()

        self.model = model

    def __init__(self):
        # shape of initial is n,800,1 [y1, y2, .. ]
        # input_shape = (800, 1)
        # shape of initial is n,800,2 [[x1, y1], .. ]
        # self.input_shape = (1600, 1)
        # self.input_shape = (1, 300) <- jak mamy tylko [[y1, y2] .. ]
        self.input_shape = (300, 2)
        # self.output_dim = 4
        self.model = None
        self.create_standard_model()
        self.model_name = "bi_lstm_model"
