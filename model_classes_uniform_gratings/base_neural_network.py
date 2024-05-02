import keras
from keras import Input
from keras.models import Sequential
from keras.src.layers import Dropout
from tensorflow.keras.layers import Flatten, Dense
from keras.src.layers import Flatten, Activation
from keras import activations


class BaseNeuralNetwork:

    def create_standard_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            Dense(400),
            Activation(activation=activations.relu),
            Dense(200),
            Activation(activation=activations.relu),
            Dense(50),
            Activation(activation=activations.relu),
            Dropout(0.5),
            Flatten(),
            Dense(self.output_dim, activation='linear')
        ])

        root_mean_squared_error = keras.metrics.RootMeanSquaredError()
        mean_squared_error = keras.metrics.MeanSquaredError()
        mean_absolute_error = keras.metrics.MeanAbsoluteError()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-7),
                      metrics=[root_mean_squared_error, mean_squared_error,
                               mean_absolute_error],
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
        self.model_name = "base_nn_model"
