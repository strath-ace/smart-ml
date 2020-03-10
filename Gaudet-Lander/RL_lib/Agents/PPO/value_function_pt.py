"""
State-Value Function

Modified from  code Written by Patrick Coady (pat-coady.github.io)
added option for value function clipping

"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
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
        self.vf_net = VF_net(obs_dim,network_scale)
        self.optimizer = torch.optim.Adam(self.vf_net.parameters(), self.vf_net.lr)  
        print('Value Function: cliprange = ',self.cliprange, ' network_scale = ',self.network_scale)
 
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
        y_hat = self.vf_net.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        old_vpred = self.vf_net.predict(x_train)
        for e in range(self.epochs):
            x_train, y_train, old_vpred = shuffle(x_train, y_train, old_vpred)
            for j in range(num_batches):
                self.optimizer.zero_grad()
                start = j * batch_size
                end = (j + 1) * batch_size
                vpred = self.vf_net.forward(x_train[start:end, :])
                loss = self.get_loss(vpred, torch.from_numpy(old_vpred[start:end]).float(), torch.from_numpy(y_train[start:end]).float())
                loss.backward()
                self.optimizer.step()
        y_hat = self.predict(x)
        ev_loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func
        self.exp_var_stat = exp_var
        logger.log({'ValFuncLoss': ev_loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

    def get_loss(self,pred,old_pred,targ):
        if self.cliprange is not None:
            vpred_clipped = old_pred + torch.clamp(pred - old_pred, -self.cliprange, self.cliprange)
            error = pred - targ
            loss1 = torch.mul(error, error)
            error = vpred_clipped - targ 
            loss2 = torch.mul(error, error)
            loss = 0.5 * torch.mean(torch.max(loss1,loss2))
        else:
            error = pred - targ
            loss = torch.mean(error, error)
        return loss

    def predict(self,x):
        return self.vf_net.predict(x)
     
class VF_net(nn.Module):
    def __init__(self, obs_dim, network_scale):
        super(VF_net, self).__init__()
        hid1_size = obs_dim * network_scale  # 10 chosen empirically on 'Hopper-v1'
        hid3_size = 5  
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr =  1e-2 / np.sqrt(hid2_size)
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, 1)
        self.apply(self.weights_init)

    def weights_init(self,m):
        if isinstance(m,nn.Linear):
            print('layer, size ',m, m.weight.data.size()[1])
            torch.nn.init.normal_(m.weight.data,std=1/np.sqrt(m.weight.data.size()[1]))
            print('got it')

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return torch.squeeze(x)

    def predict(self,x, grad=False):
        if grad:
            y = self.forward(x)
        else:
            with torch.no_grad():
                y = self.forward(x).numpy()
        return y


 
