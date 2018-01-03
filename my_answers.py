import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import string
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):

    series_len = len(series)
    in_out_pairs = series_len - window_size

    # containers for input/output pairs
    X = []
    for i in range(in_out_pairs):
        X.append(series[i:i + window_size])

    y = []
    y = series[window_size:]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):

    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1, activation='tanh'))

    return model


# TODO: return the text input with only ascii lowercase and the punctuation
# given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    alphabet = list(string.ascii_lowercase)
    allowed_chars = set(punctuation + alphabet + [' '])

    all_chars = set(text)
    chars_to_remove = all_chars - allowed_chars

    for char_to_remove in chars_to_remove:
        text = text.replace(char_to_remove, ' ')

    return text


# TODO: fill out the function below that transforms the input text and
# window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):

    text_len = len(text)
    in_out_pairs = text_len - window_size

    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, in_out_pairs, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])

    return inputs, outputs


# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation,
# categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):

    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))

    return model
