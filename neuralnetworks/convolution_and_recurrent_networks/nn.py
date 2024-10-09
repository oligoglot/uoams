"""
The main code for the recurrent and convolutional networks assignment.

See README.md for details.
"""

from typing import Tuple, List, Dict

import tensorflow

def create_toy_rnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a recurrent neural network for a toy problem.

    The network will take as input a sequence of number pairs, (x_{t}, y_{t}),
    where t is the time step. It must learn to produce x_{t-3} - y{t} as the
    output of time step t.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.SimpleRNN(64, input_shape=input_shape, return_sequences=True),
        tensorflow.keras.layers.LSTM(32, return_sequences=True),
        tensorflow.keras.layers.Dense(16, activation='relu'),
        tensorflow.keras.layers.Dense(n_outputs)
    ])

    model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.01),
                  loss='mse',
                  metrics=['mae'])

    fit_kwargs = {
        'batch_size': 32,
        'epochs': 100,
        'validation_split': 0.2,
        'callbacks': [
            tensorflow.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        ]
    }

    return model, fit_kwargs

def create_mnist_cnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a convolutional neural network for digit classification.

    The network will take as input a 28x28 grayscale image, and produce as
    output one of the digits 0 through 9. The network will be trained and tested
    on a fraction of the MNIST data: http://yann.lecun.com/exdb/mnist/

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tensorflow.keras.layers.BatchNormalization(),
        tensorflow.keras.layers.MaxPool2D((2, 2)),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dense(n_outputs, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    fit_kwargs = {
        'batch_size': 32,
        'epochs': 10,
        'validation_split': 0.2,
        'callbacks': [
            tensorflow.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        ]
    }

    return model, fit_kwargs

def create_youtube_comment_rnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a recurrent neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    vocab_size = len(vocabulary)

    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Embedding(vocab_size, 100, input_length=None),
        tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(64, return_sequences=True)),
        tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(32)),
        tensorflow.keras.layers.Dense(16, activation='relu'),
        tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    fit_kwargs = {
        'batch_size': 32,
        'epochs': 10,
        'validation_split': 0.2,
        'callbacks': [tensorflow.keras.callbacks.EarlyStopping(patience=2)]
    }

    return model, fit_kwargs

def create_youtube_comment_cnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a convolutional neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    vocab_size = len(vocabulary)

    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Embedding(vocab_size, 100, input_length=None),
        tensorflow.keras.layers.Conv1D(128, 5, activation='relu'),
        tensorflow.keras.layers.GlobalMaxPool1D(),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dense(n_outputs, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    fit_kwargs = {
        'batch_size': 64,
        'epochs': 10,
        'validation_split': 0.2,
        'callbacks': [tensorflow.keras.callbacks.EarlyStopping(patience=2)]
    }

    return model, fit_kwargs
