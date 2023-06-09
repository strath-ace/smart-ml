# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2021 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------
"""Q-Networks for calculating action-value estimates"""

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf
    
import random
import numpy as np
import pdb


class MLPQNet():
    """
    A Q-Network with a MultiLayer Perceptron structure

    ...

    Attributes
    ----------
    n_layer : int
        Number of hidden layers
    s_input : tf.Tensor
        Placeholder for inputs to the network
    w : list of tf.Variable
        Input weights to each hidden layer
    b : list of tf.Variable
        Biases for each hidden node
    W : tf.Variable
        Output weights
    layers : list of tf.Tensor
        List containing the state input plus each hidden layer operation
    Q_est : tf.Tensor
        Estimated action values calculated from the network
    var_list : list of tf.Variable
        All updatable parameters in the network
    p_dict : dict
        Dictionary referring to all updatable parameters
    new_params : dict
        Placeholders for parameters to assign to the network
    p_assign : list of tf.Tensor
        Assignment operations for new_params
    target : bool
        Indicates if network is a target network
    prep_state : tf.Tensor or None
        Tensor used as a preprocessed state - not used for MLPQNet
    k : int
        Minibatch size for updats
    nextQ : tf.Tensor
        Placeholder for new action-value estimates used to update
    updateModel : tf.Operation
        For running network updates
    sess : tf.Session
        Tensorflow session
        
    Methods
    -------
    assign_params(self,p_new)
        asdf
    get_params(self)
        asdf
    Q_predict(self, s=None, s_prep=None)
        asdf
    update(self, S, Q)
        asdf
    """
    def __init__(self, state_size, action_size,
                 hidden_layers=[20, 10], alpha=0.01, activation_function='tanh', update_steps=50, clip_norm=1.0,
                 W_init_magnitude=0.1, w_init_magnitude=0.1, b_init_magnitude=0.0, minibatch_size=10,
                 is_target=False, seed=None, **kwargs):
        """
        Parameters
        ----------
        state_size, action_size : int
            Size of the environment state space and action space
        hidden_layers : array of int, optional
            Number of hidden nodes in each layer, default [20, 10]
        alpha : float, optional
            Learning rate, default 0.01
        activation_function : str, optional
            Neuron activation function, default tanh
        w_mag, b_mag, W_mag : float, optional
            Initialisation magnitude for each set of parameters, default 0.1, 0.0, 0.1
        is_target : bool, optional
            Whether the network is a target network which does not update, default False
        seed : int, optional
            Seed for random number generation, default None
        """
        # Build network
        self.n_layer = len(hidden_layers)
        self.s_input = tf.placeholder(shape=[None,state_size],dtype=tf.float32)
        self.w = []
        self.b = []
        N_hid = hidden_layers.copy()
        N_hid.insert(0,state_size)
        for i in range(self.n_layer):
            self.w.append(tf.Variable(tf.random_uniform([N_hid[i],N_hid[i+1]],0,w_init_magnitude,seed=seed)))
            self.b.append(tf.Variable(tf.random_uniform([1,N_hid[i+1]],0,b_init_magnitude,seed=seed)))
        self.W = tf.Variable(tf.random_uniform([N_hid[-1],action_size],0,W_init_magnitude,seed=seed))

        try:
            act_fn = tf.keras.activations.get(activation_function)
        except ValueError:
            raise ValueError('Invalid activation function: \'{}\''.format(activation_function))
        self.layers = [self.s_input]
        for w, b in zip(self.w,self.b):
            self.layers.append(act_fn(tf.add(tf.matmul(self.layers[-1],w),b)))
        self.Q_est = tf.matmul(self.layers[-1],self.W)
        self.var_list = self.w + self.b + [self.W]

        self.p_dict = {'W':self.W, 'w':self.w, 'b':self.b}
        self.new_params = {'W':tf.placeholder(shape=[None,None],dtype=tf.float32)}
        self.new_params['w'] = [tf.placeholder(shape=[None,None],dtype=tf.float32) for _ in self.w]
        self.new_params['b'] = [tf.placeholder(shape=[None,None],dtype=tf.float32) for _ in self.b]

        self.p_assign = [self.W.assign(self.new_params['W'])]
        self.p_assign += [w.assign(self.new_params['w'][i]) for i, w in enumerate(self.w)]
        self.p_assign += [b.assign(self.new_params['b'][i]) for i, b in enumerate(self.b)]

        self.target=is_target
        if self.target:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            return

        # Update rules
        self.prep_state = None
        self.k = int(minibatch_size)
        self.nextQ = tf.placeholder(shape=[None,action_size],dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Q_est))
        trainer = tf.train.RMSPropOptimizer(alpha)
        if clip_norm is not None:
            grads = trainer.compute_gradients(loss,self.var_list)
            cap_grads = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grads]
            self.updateModel = trainer.apply_gradients(cap_grads)
        else:
            self.updateModel = trainer.minimize(loss,var_list=self.var_list)

        # Start tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def assign_params(self,p_new):
        """Assign new values p_new to updatable parameters"""
        p_assign_dict = {self.new_params['W']:p_new['W']}
        for n in range(self.n_layer):
            p_assign_dict[self.new_params['w'][n]] = p_new['w'][n]
            p_assign_dict[self.new_params['b'][n]] = p_new['b'][n]
        self.sess.run(self.p_assign, feed_dict=p_assign_dict)

    def get_params(self):
        """Return current values of updatable parameters"""
        return self.sess.run(self.p_dict)

    def Q_predict(self, s=None, s_prep=None):
        """
        Return predicted action-values for the state

        ...

        Parameters
        ----------
        s : array-like, optional
            State in its `normal` form
        s_prep : array_like, optional
            State in a preprocessed form defined by `prep_state`

        Returns
        -------
        array-like
            Predicted action values for the state, empty if `s` and `s_prep` are `None`
        """
        if s is not None:
            return self.sess.run(self.Q_est,feed_dict={self.s_input:s})
        elif s_prep is not None:
            return self.sess.run(self.Q_est,feed_dict={self.prep_state:s_prep})
        else:
            return []

    def update(self, S, Q):
        """Updates based on states and target action values"""
        if not self.target:
            self.sess.run(self.updateModel,{self.s_input:S,self.nextQ:Q})
