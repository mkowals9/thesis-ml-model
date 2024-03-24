import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.src.layers import Reshape


class LstmModel:

    def create_standard_model(self):

        model = Sequential([
            LSTM(300, activation='relu', return_sequences=True, input_shape=self.input_shape),
            # Dropout(rate=0.5),
            LSTM(200, activation='relu', return_sequences=True),
            # Dropout(rate=0.2),
            LSTM(200, activation='relu', return_sequences=False),
            Dense(150, activation='relu'),
            Dense(15, activation='relu'),
            Dense(15, activation='linear'),
        ])

        mean_squared_error = keras.metrics.MeanSquaredError()
        mean_absolute_error = keras.metrics.MeanAbsoluteError()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-6),
                      metrics=[mean_squared_error, mean_absolute_error],
                      loss='mean_squared_error')
        model.summary()

        self.model = model

    def __init__(self):
        # shape of initial is n,800,1 [y1, y2, .. ]
        # input_shape = (800, 1)
        # shape of initial is n,800,2 [[x1, y1], .. ]
        # self.input_shape = (1600, 1)
        self.input_shape = (1, 300,)
        # self.output_dim = 4
        self.model = None
        self.create_standard_model()
        self.model_name = "lstm_model"
