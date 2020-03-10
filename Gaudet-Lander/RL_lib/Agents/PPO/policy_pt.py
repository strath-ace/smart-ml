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
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Action_converter
import utils
import sklearn.utils 

class Policy(object):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim, actions_per_dim=3, kl_targ=0.003,epochs=20,discretize=False,
                 network_scale=10,test_mode=False,shuffle=True,
                 entropy_coeff=0.0, servo_kl=False, beta=0.1):
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
            servo_kl:               boolean:  set to False to not servo beta to KL, which is original PPO implementation
            beta:                   clipping parameter for pessimistic loss ratio
 
        """
        print('PPO Policy pytorch')
        self.servo_kl = servo_kl
        self.network_scale = network_scale
        self.test_mode = test_mode
        self.entropy_coeff = entropy_coeff 
        self.discretize = discretize
        self.shuffle = shuffle
        self.actions_per_dim = actions_per_dim
        self.kl_stat = None
        self.entropy_stat = None
        self.kl_targ = kl_targ
        self.epochs = epochs 
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_beta = 0.5
        self.min_beta = 0.01 
        self.beta = beta
        self.action_converter = Action_converter(1,actions_per_dim)

        self.policy_net = Policy_net(obs_dim, act_dim, network_scale)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), self.policy_net.lr)
        print('Actor Test Mode: ',self.test_mode)
        print('clip param: ',self.beta)



    def _kl_entropy(self, logp, old_logp, log_vars):
        """

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """

        kl = 0.5 * np.mean((logp - old_logp)**2)
        
        entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              np.sum(log_vars))

        return kl, entropy


    def sample(self, obs):
        """Draw sample from policy distribution"""

        deterministic_action, log_vars = self.policy_net.predict(obs)

        if self.test_mode:
            action = deterministic_action 
        else:
            #print('log_vars: ',log_vars)
            sd = np.exp(log_vars / 2.0)
            #print('sd: ', sd)
            action = deterministic_action + np.random.normal(scale=sd)

        if self.discretize:
            idx = self.action_converter.action2idx(action[0])
            discrete_action = self.action_converter.idx2action(idx)
            env_action = discrete_action
        else:  
            env_action = action
        #print('env act: ',env_action)
        return action, env_action


    def update(self, observes, actions, advantages, logger):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            logger: Logger object, see utils.py
        """

        actions_t = torch.from_numpy(actions).float()
        advantages_t = torch.from_numpy(advantages).float()

        with torch.no_grad():
            means, logvars = self.policy_net.forward(observes)
        old_logp, _ = self.calc_logp(actions_t, means, logvars)   
        old_logp_np = old_logp.detach().numpy() 
        loss, kl, entropy = 0, 0, 0
 
        for e in range(self.epochs):

            if self.shuffle:
                    observes, actions, advantages, old_logp_np = sklearn.utils.shuffle(observes,actions,advantages,old_logp_np)

            actions_t = torch.from_numpy(actions).float()

            self.optimizer.zero_grad()
            means, log_vars_tmp = self.policy_net.forward(observes)
            logp, log_vars = self.calc_logp(actions_t, means, log_vars_tmp)
            loss = self.calc_loss(logp, torch.from_numpy(old_logp_np).float(), torch.from_numpy(advantages).float(), self.beta)
            loss.backward()
            self.optimizer.step()

            self.log_vars_np = log_vars.detach().numpy()
            kl, entropy = self._kl_entropy(logp.detach().numpy(), old_logp_np, self.log_vars_np)

            if kl > 4.0 * self.kl_targ and self.servo_kl:
                print(' *** BROKE ***')
                break 

        if self.servo_kl:
            self.adjust_beta(kl)
        for g in self.optimizer.param_groups:
            g['lr'] = self.policy_net.lr * self.lr_multiplier
 
        print('kl = ',kl, ' beta = ',self.beta,' lr_mult = ',self.lr_multiplier)
        self.kl_stat = kl
        self.entropy_stat = entropy
        var_monitor = np.exp(self.log_vars_np/2.0)
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

    def calc_loss(self,logp, old_logp, advantages, beta):
        ratio = torch.exp(logp - old_logp)
        surr1 = advantages * ratio
        surr2 = advantages * torch.clamp(ratio, 1.0 - beta, 1.0 + beta)
        
        loss = -torch.mean(torch.min(surr1,surr2)) 
        return loss

    def calc_logp(self, act, means, log_vars):
        log_vars = torch.sum(log_vars, 0) - 1.0
        logp1 = -0.5 * torch.sum(log_vars)
        diff = act - means
        logp2 = -0.5 * torch.sum(torch.mul(diff, diff) / torch.exp(log_vars), 1)
        logp3 = -0.5 * np.log(2.0 * np.pi) * self.act_dim
        logp = logp1 + logp2 + logp3
        return logp, log_vars

class Policy_net(nn.Module):
    def __init__(self, obs_dim, act_dim, network_scale):
        super(Policy_net, self).__init__()
        hid1_size = obs_dim * network_scale  
        hid3_size = act_dim * network_scale  
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = 9e-4 / np.sqrt(hid2_size) 
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)
        self.apply(self.weights_init)

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))
 
    def weights_init(self,m):
        if isinstance(m,nn.Linear):
            #print('layer, size ',m, m.weight.data.size()[1])
            torch.nn.init.normal_(m.weight.data,std=1/np.sqrt(m.weight.data.size()[1]))

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x, self.log_vars  

    def predict(self,x, return_tensor=False):
        with torch.no_grad():
            if return_tensor:
                y, log_vars = self.forward(x)
                log_vars = torch.sum(log_vars, 0) - 1.0
            else:
                y, log_vars = self.forward(x)
                log_vars = (torch.sum(log_vars, 0) - 1.0).numpy()
                y = y.numpy()
        return y, log_vars

