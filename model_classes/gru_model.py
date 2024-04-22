import keras
from keras import Input, regularizers
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.src.layers import Dropout, Reshape, Flatten, Activation
from keras import activations


class GruModel:

    def create_standard_model(self):
        l2_lambda = 0.001
        l1_lambda = 0.001

        model = Sequential([
            Input(shape=self.input_shape),
            GRU(50, return_sequences=True),
            Activation(activations.relu),
            Dropout(rate=0.25),
            # GRU(40, return_sequences=True),
            # Activation(activations.relu),
            # Dropout(rate=0.5),
            Dense(30),
            Activation(activations.relu),
            Dense(self.output_dim + 10, kernel_regularizer=regularizers.l1(l1_lambda)),
            Activation(activations.relu),
            Dense(self.output_dim, kernel_regularizer=regularizers.l2(l2_lambda)),
            Activation(activations.relu),
            Reshape((-1, self.output_dim)),
            Flatten(),
            Dense(self.output_dim),
            Activation(activations.linear),
            Dense(self.output_dim),
            Activation(activations.relu),
        ])

        mean_squared_error = keras.metrics.MeanSquaredError()
        mean_absolute_error = keras.metrics.MeanAbsoluteError()
        root_mean_squared_error = keras.metrics.RootMeanSquaredError()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-7),
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
        self.output_dim = 15
        self.model = None
        self.create_standard_model()
        self.model_name = "gru_model"
