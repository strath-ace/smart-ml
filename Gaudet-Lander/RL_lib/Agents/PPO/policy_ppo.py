"""
    Implements PPO

    PPO: https://arxiv.org/abs/1707.06347
    Modified from policy Written by Patrick Coady (pat-coady.github.io) to implement
    latest version of PPO with pessimistic ratio clipping

    o Has an option to servo both the learning rate and the clip_param to keep KL 
      within  a specified range. This helps on some control tasks
      (i.e., Mujoco Humanid-v2)
 
    o Uses approximate KL 

    o Models distribution of actions as a Gaussian with variance not conditioned on state

    o Has option to discretize sampled actions
 
"""
import numpy as np
import tensorflow as tf
from utils import Action_converter
import utils
import sklearn.utils 

class Policy(object):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim, actions_per_dim=2, kl_targ=0.003,epochs=20,discretize=False,
                 input_network_scale=10,output_network_scale=10,test_mode=False,shuffle=True,
                 entropy_coeff=0.0,eta=0.0, servo_kl=False, beta=0.1, lr=None):
        """
        Args:
            obs_dim:                num observation dimensions (int)
            act_dim:                num action dimensions (int)
            actions_per_dim:        used when discretizing action space
            kl_targ:                target KL divergence between pi_old and pi_new
            epochs:                 number of epochs per update
            discretize:             boolean, True discretizes action space
            input_network_scale:    NN input layer is of dim <input_network_scale> * obs_dim
            output_network_scale:   NN layer prior to output layer is of dim <input_network_scale> * act_dim
            test_mode:              boolean, True removes all exploration noise
            shuffle:                boolean, shuffles data each epoch                   
            entropy_coeff:          adds a loss term that encourages exploration.  This almost always makes things
                                    worse for control tasks
            eta:                    KL hingle loss parameter
            servo_kl:               boolean:  set to False to not servo beta to KL, which is original PPO implementation
            beta:                   clipping parameter for pessimistic loss ratio
 
        """
        print('PPO Policy 1')
        self.servo_kl = servo_kl
        self.input_network_scale = input_network_scale
        self.output_network_scale = output_network_scale
        self.test_mode = test_mode
        self.entropy_coeff = entropy_coeff 
        self.discretize = discretize
        self.shuffle = shuffle
        self.actions_per_dim = actions_per_dim
        self.kl_stat = None
        self.entropy_stat = None
        self.eta = eta  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = kl_targ
        self.epochs = epochs 
        self.lr = lr 
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_beta = 0.5
        self.min_beta = 0.01 
        self.beta = beta
        self._build_graph()
        self._init_session()
        self.action_converter = Action_converter(1,actions_per_dim)
        print(self.input_network_scale)
        print('Actor Test Mode: ',self.test_mode)
        print('clip param: ',self.beta)

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._sample()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        # strength of D_KL loss terms:
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'eta')
        # log_vars and means with pi_old (previous step's policy parameters):
        self.old_logp_ph =  tf.placeholder(tf.float32, (None,), 'old_logp')

        #self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        #self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _policy_nn(self):
        """ Neural net for policy approximation function

        Policy parameterized by Gaussian means and variances. NN outputs mean
         action based on observation. Trainable variables hold log-variances
         for each action dimension (i.e. variances not determined by NN).
        """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        hid1_size = self.obs_dim * self.input_network_scale  # 10 empirically determined
        hid3_size = self.act_dim * self.output_network_scale  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        if self.lr is None:
            self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined
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
        self.means = tf.layers.dense(out, self.act_dim,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / hid3_size)), name="means")
        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_size) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) - 1.0

        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_size, hid2_size, hid3_size, self.lr, logvar_speed))

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        logp += -0.5 * np.log(2.0 * np.pi) * self.act_dim 
        self.logp = logp

    def _kl_entropy(self):
        """
        Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """

        self.kl = 0.5 * tf.reduce_mean(tf.square(self.logp - self.old_logp_ph))
        
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))

    def _sample(self):
        """ Sample from distribution, given observation """
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) *
                            tf.random_normal(shape=(self.act_dim,)))

        self.deterministic_act = (self.means)
       
 

    def _loss_train_op(self):
        ratio = tf.exp(self.logp - self.old_logp_ph)
        surr1 = self.advantages_ph * ratio
        surr2 = self.advantages_ph * tf.clip_by_value(ratio, 1.0 - self.beta_ph, 1.0 + self.beta_ph)

        loss1 = -tf.reduce_mean(tf.minimum(surr1,surr2)) - self.entropy_coeff * self.entropy
        loss2 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))

        self.loss = loss1 + loss2
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, observes, actions, advantages, old_logp_np):

        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.old_logp_ph: old_logp_np,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier}

        _, loss, kl, entropy, log_var_monitor = self.sess.run([self.train_op, self.loss, self.kl, self.entropy, self.log_vars], feed_dict)

        return loss, kl, entropy, log_var_monitor 

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        if self.test_mode:
            action = self.sess.run(self.deterministic_act , feed_dict=feed_dict)
        else:
            action = self.sess.run(self.sampled_act , feed_dict=feed_dict)

        if self.discretize:
            idx = self.action_converter.action2idx(action[0])
            discrete_action = self.action_converter.idx2action(idx)
            env_action = discrete_action
        else:  
            env_action = action
#         print('env act: ',env_action)
        return action, env_action


    def update(self, observes, actions, advantages, logger):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            logger: Logger object, see utils.py
        """

        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages}
        old_logp_np = np.squeeze(self.sess.run([self.logp],feed_dict))

        loss, kl, entropy = 0, 0, 0
 
        for e in range(self.epochs):

            if self.shuffle:
                    observes, actions, advantages, old_logp_np = sklearn.utils.shuffle(observes,actions,advantages,old_logp_np)


            loss, kl, entropy, log_var_monitor = self.train(observes, actions, advantages, old_logp_np)

            if kl > 4.0 * self.kl_targ and self.servo_kl:
                print(' *** BROKE ***')
                break 

        if self.servo_kl:
            self.adjust_beta(kl)
        print('kl = ',kl, ' beta = ',self.beta,' lr_mult = ',self.lr_multiplier)
        self.kl_stat = kl
        self.entropy_stat = entropy
        var_monitor = np.exp(log_var_monitor/2.0)
        print('var: ' ,var_monitor),
        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    'Variance' : np.max(var_monitor),
                    'lr_multiplier': self.lr_multiplier})

    def adjust_beta(self,kl):
        if  kl < self.kl_targ / 2:
            self.beta = np.minimum(self.max_beta, 1.5 * self.beta)  # max clip beta
            #print('too low')
            if self.beta > (self.max_beta/2) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5
        elif kl > self.kl_targ * 2:
            #print('too high')
            self.beta = np.maximum(self.min_beta, self.beta / 1.5)  # min clip beta
            if self.beta <= (2*self.min_beta) and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5


    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()
