# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------ Copyright (C) 2023 University of Strathclyde and Author ------
# ---------------- Author: Francesco Marchetti ------------------------
# ----------- e-mail: francesco.marchetti@strath.ac.uk ----------------

# Alternatively, the contents of this file may be used under the terms
# of the GNU General Public License Version 3.0, as described below:

# This file is free software: you may copy, redistribute and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3.0 of the License, or (at your
# option) any later version.

# This file is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from copy import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential


def create_NN_model(n_input, n_hidden_layers, nodes_per_hidden_layer, n_output, activation):
    """
    Function used to create the NN model

    Attributes:
        n_input: int
            number of input of the NN
        n_hidden_layers: int
            number of hidden layers
        nodes_per_hidden_layer: list
            number of nodes per hidden layer
        n_output: int
            number of output neurons
        activation: str
            activation function. To be chosen among those implemented in tensorflow

    Return:
        model: tensorflow model
            NN model
        tot_weights_n_biases: int
            number of weights and biases
    """
    model = Sequential()
    initializer = tf.keras.initializers.Zeros()#GlorotNormal(seed=obj.NNseed)

    model.add(tf.keras.layers.Dense(nodes_per_hidden_layer[0], input_shape=(n_input,), kernel_initializer=initializer,
                                      bias_initializer=initializer,
                                      activation=activation, name='Hidden_1', dtype='float64'))  # input layer and first hidden layer
    for i in range(1, n_hidden_layers):
        model.add(tf.keras.layers.Dense(nodes_per_hidden_layer[i], kernel_initializer=initializer,
                                          bias_initializer=initializer,
                                          activation=activation, name='Hidden_{}'.format(i+1), dtype='float64'))  # second hidden layer
    model.add(tf.keras.layers.Dense(n_output, activation='linear', kernel_initializer=initializer,
                                      bias_initializer=initializer, name='Output', dtype='float64'))  # output layer

    tot_weights_n_biases = 0
    for i in range(len(model.get_weights())):
        tot_weights_n_biases += model.get_weights()[i].size

    return model, tot_weights_n_biases

def update_NN_weights(NNmodel, new_weights, n_weights):
    """
    Function used to update the weights in the NN models

    Attributes:
        NNmodel: tensorflow model
            NN model
        new_weights: array
            updated values of weights
        n_weights: int
            number of weights

    Return:
        NNmode: tensorflow model
            updated NN model
    """
    new_weights_reshaped = []
    tot_len_old = 0

    tuned_weights = np.reshape(new_weights, (n_weights, 1))
    for i in range(len(NNmodel.weights)):
        shape = np.shape(NNmodel.get_weights()[i])
        if shape.__len__() == 2:
            tot_len = shape[0] * shape[1]
        else:
            tot_len = shape[0]
        new_weights_reshaped.append(copy(np.reshape(tuned_weights[tot_len_old:tot_len_old + tot_len], shape)))
        tot_len_old += tot_len
    NNmodel.set_weights(new_weights_reshaped)
    return NNmodel