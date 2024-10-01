"""
The main code for the feedforward networks assignment.
See README.md for details.
"""
from typing import Tuple, Dict
import tensorflow as tf
from tensorflow import keras

def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[keras.models.Model,
keras.models.Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters and the same activation functions.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """

    # Deep network
    deep_model = keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(n_inputs,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(n_outputs)
    ])

    # Wide network
    wide_model = keras.models.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(n_inputs,)),
        keras.layers.Dense(n_outputs)
    ])

    return deep_model, wide_model


def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[keras.models.Model,
keras.models.Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """

    def create_base_model():
        return keras.models.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(n_inputs,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(n_outputs, activation='softmax')
        ])

    # Model with dropout
    dropout_model = create_base_model()
    dropout_model.layers.insert(1, keras.layers.Dropout(0.3))
    dropout_model.layers.insert(3, keras.layers.Dropout(0.3))
    dropout_model.layers.insert(5, keras.layers.Dropout(0.3))

    # Model without
