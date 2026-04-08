"""
Discontinuous NNs

Adapted from https://github.com/Fra0013To/deltaNN
"""

####################################################################################
# MIT License
#
# Copyright (c) 2022 Francesco Della Santa
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
####################################################################################

import tensorflow as tf
from tensorflow.keras.layers import Dense

def tf_heaviside(tensor, dtype=tf.float32):
    """
    Function that returns the tensor H given by the element-wise application of the Heaviside function.
    We compute the Heaviside function H(x) = sign(relu(x) + sign(x) + 1).
    In this way, we obtain the Heaviside function: H(x) = 1 for each x>=0, H(x)=0 otherwise.
    In this way, tensorflow can compute the derivative of H using the derivatives of relu and sign functions.

    :param tensor: TF tensor
    :param dtype: output type (default tf.float32)
    :return H: element-wise Heaviside function applied to input tensor
    """
    H = tf.sign(tf.keras.activations.relu(tensor) + tf.sign(tensor) + 1)
    H = tf.cast(H, dtype=dtype)

    return H

class DiscontinuityDense(Dense):
    """
    New subclass of tf.keras.layers.Dense s.t. it has a trainable discontinuity in 0.
    The map of this layer is:

    L(x) = f(W'x + b) + eps * H(W'x + b),

    where:
    f: activation function;
    W: layer's weight matrix
    b: layer's bias vector
    eps: layer's vector of trainable discontinuities
    """
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 discontinuity_initializer='zeros',
                 **options):
        """
        Initialization method
        :param units: same of class Dense
        :param activation: same of class Dense
        :param use_bias: same of class Dense
        :param kernel_initializer: same of class Dense
        :param bias_initializer: same of class Dense
        :param kernel_regularizer: same of class Dense
        :param bias_regularizer: same of class Dense
        :param activity_regularizer: same of class Dense
        :param kernel_constraint: same of class Dense
        :param bias_constraint: same of class Dense
        :param discontinuity_initializer: NEW PARAMETER, keras initializer for trainable discontinuities (def. Zeros())
        :param options: same of class Dense
        """
        super(DiscontinuityDense, self).__init__(units, activation, use_bias,
                                                 kernel_initializer, bias_initializer,
                                                 kernel_regularizer, bias_regularizer, activity_regularizer,
                                                 kernel_constraint, bias_constraint,
                                                 **options)

        if type(discontinuity_initializer) is str:
            discontinuity_initializer = getattr(tf.keras.initializers, discontinuity_initializer)

        if type(discontinuity_initializer) == type:
            discontinuity_initializer = discontinuity_initializer()

        self.discontinuity_initializer = discontinuity_initializer

    def get_config(self):

        config = super(DiscontinuityDense, self).get_config()
        config['discontinuity_initializer'] = self.discontinuity_initializer

        return config

    def build(self, input_shape):

        # Kernel of classical Dense layer
        # self.kernel = self.add_variable("kernel",
        #                                 shape=[int(input_shape[-1]),
        #                                        self.num_outputs])

        # Kernel creation
        self.kernel = self.add_weight('kernel', shape=[int(input_shape[-1]), self.units],
                                      initializer=self.kernel_initializer)
        self.bias = self.add_weight('bias', shape=[self.units],
                                    initializer=self.bias_initializer)
        self.discontinuity = self.add_weight('discontinuity', shape=[self.units],
                                             initializer=self.discontinuity_initializer)

    def call(self, input):
        matmul_tensor = tf.matmul(input, self.kernel)

        matmul_bias_tensor = tf.add(matmul_tensor, self.bias)

        add_discontinuty_tensor = tf.multiply(tf_heaviside(matmul_bias_tensor), self.discontinuity)

        out_tensor = tf.add(self.activation(matmul_bias_tensor), add_discontinuty_tensor)

        return out_tensor

