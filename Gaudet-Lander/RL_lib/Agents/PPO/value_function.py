"""
State-Value Function

Modified from  code Written by Patrick Coady (pat-coady.github.io)
added option for value function clipping

"""

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class Value_function(object):
    """ NN-based state-value function """
    def __init__(self, obs_dim, epochs=20, cliprange=0.2, network_scale=10, lr=None):
        """
        Args:
            obs_dim:        number of dimensions in observation vector (int)
            network_scale:  NN input layer is of dim <input_network_scale> * obs_dim
            epochs:         number of epochs per update
            cliprange:      for limiting value function updates
 
        """
        self.cliprange = cliprange
        self.network_scale = network_scale
        self.exp_var_stat = None
        self.epochs = epochs
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.lr =  lr  
        self._build_graph()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)
        print('Value Function: cliprange = ',self.cliprange, ' network_scale = ',self.network_scale)
 
    def _build_graph(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            self.oldval_ph = tf.placeholder(tf.float32, (None,), 'old_val_valfunc')

            # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
            hid1_size = self.obs_dim * self.network_scale  # 10 chosen empirically on 'Hopper-v1'
            hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
            if self.lr is None:
                self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined
            print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
                  .format(hid1_size, hid2_size, hid3_size, self.lr))
            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid3_size)), name='output')
            self.out = tf.squeeze(out)
            if self.cliprange is not None:
                vpred_clipped = self.oldval_ph + tf.clip_by_value(self.out - self.oldval_ph, -self.cliprange, self.cliprange)
                loss1 = tf.square(self.out - self.val_ph)
                loss2 = tf.square(vpred_clipped - self.val_ph)
                self.loss = 0.5 * tf.reduce_mean(tf.maximum(loss1,loss2))
            else:
                self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def fit(self, x, y, logger):
        print('FIT: ',np.max(np.abs(y)), np.max(np.abs(x)), np.mean(np.abs(y)), np.mean(np.abs(x)))
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        old_vpred = self.predict(x_train)
        for e in range(self.epochs):
            x_train, y_train, old_vpred = shuffle(x_train, y_train, old_vpred)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end],
                             self.oldval_ph: old_vpred[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func
        self.exp_var_stat = exp_var
        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

    def predict(self, x):
        """ Predict method """
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()
