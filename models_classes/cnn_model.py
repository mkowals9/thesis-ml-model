import keras
from keras import Input
from keras.models import Sequential
from keras.src.layers import Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Reshape


class CnnModel:

    def create_standard_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            Reshape((500, 1)),
            Conv1D(filters=30, kernel_size=2, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=28, kernel_size=2, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=25, kernel_size=2, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=20, kernel_size=2, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.1),
            Flatten(),
            Dense(units=self.output_dim, activation='linear')
        ])

        mean_squared_error = keras.metrics.MeanSquaredError()
        mean_absolute_error = keras.metrics.MeanAbsoluteError()
        #mean_absolute_percentage_error = keras.metrics.MeanAbsolutePercentageError()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0007, epsilon=6e-7),
                      metrics=[mean_squared_error, mean_absolute_error],
                      loss='mean_squared_error')
        model.summary()

        self.model = model

    def __init__(self):
        # shape of initial is n,800,1 [y1, y2, .. ]
        # input_shape = (800, 1)
        # shape of initial is n,800,2 [[x1, y1], .. ]
        # self.input_shape = (1600, 1)
        self.input_shape = (500, )
        self.model = None
        self.output_dim = 20
        self.create_standard_model()
        self.model_name = "bi_lstm_model"
