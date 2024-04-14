import keras
from keras import Input
from keras.models import Sequential
from keras.src.layers import Reshape, Dropout, Flatten
from keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras import regularizers


class BasicDenseModel:

    def create_standard_model(self):
        l2_lambda = 0.001

        model = Sequential([
            Input(shape=self.input_shape),
            # Bidirectional(LSTM(300, activation='relu', return_sequences=True)),
            # Dropout(rate=0.55),
            # Bidirectional(LSTM(90, activation='relu', return_sequences=True)),
            # Dropout(rate=0.55),
            # Bidirectional(LSTM(200, activation='relu', return_sequences=False)),
            Dense(300, activation='relu'),
            # Dense(250, activation='relu'),
            Dense(200, activation='softplus'),
            Dense(150, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
            # Dense(self.output_dim+100, activation='relu'),
            Dense(100, activation='relu'),
            # Dense(25, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
            # Dense(60, activation='relu'),
            # Dense(15, activation='relu'),
            # if only one array on the output
            # Dense(15, activation='linear', kernel_regularizer=regularizers.l2(l2_lambda)),
            Dense(self.output_dim, activation='relu'),
            # Dense(self.output_dim, activation='linear'),
            Reshape((300, self.output_dim)),
            Flatten(),
            Dense(self.output_dim, activation="relu"),
            Dense(self.output_dim)
        ])

        mean_squared_error = keras.metrics.MeanSquaredError()
        root_mean_squared_error = keras.metrics.RootMeanSquaredError()
        mean_absolute_error = keras.metrics.MeanAbsoluteError()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001, epsilon=1e-7),
                      metrics=[mean_squared_error, mean_absolute_error, root_mean_squared_error],
                      loss='mean_absolute_error')
        model.summary()

        self.model = model

    def __init__(self):
        # shape of initial is n,800,1 [y1, y2, .. ]
        # input_shape = (800, 1)
        # shape of initial is n,800,2 [[x1, y1], .. ]
        # self.input_shape = (1600, 1)
        # self.input_shape = (1, 300) <- when we have only [[y1, y2] .. ]
        self.input_shape = (300, 2)
        self.output_dim = 16
        self.model = None
        self.create_standard_model()
        self.model_name = "basic_dense_model"
