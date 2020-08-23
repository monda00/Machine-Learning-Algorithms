import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential


def sequential_api_model(input_shape):
    model = Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def functional_api_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


sequential_model = sequential_api_model((784,))
functional_model = functional_api_model((784,))

keras.utils.plot_model(
    sequential_model, 'sequential_model.png', show_shapes=True)
keras.utils.plot_model(
    functional_model, 'functional_model.png', show_shapes=True)
