import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.src.layers import Dropout, Reshape, Flatten


class LstmModel:

    def create_standard_model(self):
        l2_lambda = 0.001
        l1_lambda = 0.001

        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(50, activation='relu', return_sequences=True),
            Dropout(rate=0.25),
            LSTM(40, activation='relu', return_sequences=True),
            Dropout(rate=0.5),
            Dense(40, activation='relu'),
            Dense(self.output_dim+20, activation='relu'),
            Dense(self.output_dim+20, activation='relu'),
            Reshape((-1, self.output_dim)),
            Flatten(),
            Dense(self.output_dim, activation='linear'),
            Dense(self.output_dim, activation='relu'),

        ])

        mean_squared_error = keras.metrics.MeanSquaredError()
        mean_absolute_error = keras.metrics.MeanAbsoluteError()
        root_mean_squared_error = keras.metrics.RootMeanSquaredError()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-6, epsilon=1e-7),
                      metrics=[mean_squared_error, mean_absolute_error, root_mean_squared_error],
                      loss='mean_absolute_error')
        model.summary()

        self.model = model

    def __init__(self):
        # shape of initial is n,800,1 [y1, y2, .. ]
        # input_shape = (800, 1)
        # shape of initial is n,800,2 [[x1, y1], .. ]
        # self.input_shape = (1600, 1)
        self.input_shape = (300, 2)
        self.output_dim = 16
        self.model = None
        self.create_standard_model()
        self.model_name = "lstm_model"
