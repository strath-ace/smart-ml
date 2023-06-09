# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2021 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------
"""
Contains the QAgent class for training and ReplayMemory for storing 
and sampling experiences
"""

from . import networks
from random import Random
import numpy as np
import numpy.random as rand
import pdb


class ReplayMemory(list):
    """
    List for storing experiences and separately storing demonstration experience
    """
    def __init__(self,memory_size=None,demo_memory=[],n_demo=None,seed=None,**kwargs):
        list.__init__(self,[])
        self.max_len = memory_size
        self.demo_memory = demo_memory
        self.n_demo = int(n_demo) if n_demo is not None else 0
        self.rand_state = Random(seed)
    def add(self, list_add):
        self.append(list_add)
        if self.max_len:
            while len(self)>self.max_len:
                self.remove(self[0])
    def sample(self, n):
        if self.n_demo>0:
            return self.rand_state.sample(self,n-self.n_demo) + self.demo_memory.sample(self.n_demo)
        else:
            return self.rand_state.sample(self,n)


class QAgent(object):
    """
    A Q-Learning agent which uses a Q-Network to select actions

    ...

    Attributes
    ----------
    state_size : int
        Size of the environment state space
    action size : int
        Size of the environment action space
    nn : QLearn.networks
        Q-network used to approximate action-values
    nn_target : QLearn.networks
        Q-network for generating target action-values when updating
    memory : ReplayMemory
        Replay memory for storing state transitions used to update
    gamma : float
        Discount factor used for Q-learning updates
    epsilon : float
        Exploration probability that decreases linearly over training episodes
    d_eps : float
        Change in epsilon after each episode
    eps_f : float
        Final exploration probability
    f_heur : def
        Heursitic action selection for initial episodes
    n_heur : int
        Number of episodes at start of training to use f_heur
    target_steps : int
        Number of steps between target network updates
    prev_s : array-like, None
        Placeholder for storing previous state to add to memory
    prev_a : int, None
        Placeholder for storing previous action to add to memory
    ep_count : int
        Counts the number of episodes
    step_count : int
        Counts the number of steps within an episode
    rand_state : np.random.RandomState
        Random state for generating random numbers

    Methods
    -------
    action_select(self,state)
        Select action according to epsilon-greedy policy, store state and action
    action_test(self,state)
        Select action according to greedy policy
    network_update(self)
        Update nn weights using data sampled from memory
    update(self,state,reward,done,net_update=True)
        Add state transition to memory and optionally update nn weights

    """
    def __init__(self,env,net_type='MLPQNet',f_heur=None,n_heur=0,seed=None,
                 gamma=0.6,eps_i=0.9,eps_f=0.0,n_eps=400,target_steps=50,**kwargs):
        """
        Parameters
        ----------
        env : Environment
            Environment on which the agent is trained
        net_type : str, optional
            Type of nn to use, default 'MLPQNet'
        f_heur : def, optional
            Heuristic action selection for initial episodes, default None
        n_heur : int, optional
            Number of episodes to use heuristic action selection, default 0
        seed : int, optional
            Seed passed to the random number generator, default None
        gamma : float, optional
            Discount factor used for Q-learning updates, default 0.6
        eps_i : float, optional
            Initial exploration probability epsilon, default 0.9
        eps_f : float, optional
            Final exploration probability, default 0
        n_eps : int, optional
            Number of episodes to linearly decrease epsilon, default 400
        target_steps : int, optional
            Number of steps between target network updates, default 50
        **kwargs
            Additional keyword arguments passed to nn, nn_target, and memory

        Raises
        ------
        ValueError
            If net_type is not a valid QLearn.networks attribute
        """
        self.state_size = env.state_size
        self.action_size = env.action_size
        try:
            net_module = getattr(networks,net_type)
        except AttributeError:
            raise ValueError('Invalid network type: \'{}\''.format(net_type))

        self.nn = net_module(self.state_size, self.action_size, **kwargs)
        self.nn_target = net_module(self.state_size, self.action_size, 
                                    is_target=True, **kwargs)
        self.nn_target.assign_params(self.nn.get_params())

        self.memory = ReplayMemory(seed=seed,**kwargs)

        self.gamma = gamma
        self.epsilon = eps_i
        self.d_eps = (eps_i-eps_f)/float(n_eps)
        self.eps_f = eps_f
        self.f_heur = f_heur
        self.n_heur = n_heur
        self.target_steps = target_steps

        self.prev_s = None
        self.prev_a = None
        self.ep_count = 0
        self.step_count = 0
        self.rand_state = rand.RandomState(seed=seed)

    def action_select(self,state):
        """
        Select an action for the given state based on an epsilon-greedy policy

        ...

        Parameters
        ----------
        state : array-like
            State observation

        Returns
        -------
        int
            Selected action
        """
        if self.ep_count<self.n_heur and self.f_heur is not None:
            action = self.f_heur(state)
        elif self.rand_state.random()<self.epsilon:
            action = self.rand_state.randint(self.action_size)
        else:
            q_s=self.nn.Q_predict(state)
            action=np.argmax(q_s)
        self.prev_s=state
        self.prev_a=action
        return action

    def action_test(self,state):
        """
        Select an action for the given state based on a greedy policy

        ...

        Parameters
        ----------
        state : array-like
            State observation

        Returns
        -------
        int
            Selected action
        """
        q_s = self.nn.Q_predict(state)
        action = np.argmax(q_s)
        return action

    def network_update(self):
        """Update nn weights and periodically update target_nn"""
        D_update = self.memory.sample(self.nn.k)
        s, a, r, Sd, St = (np.stack([d[0] for d in D_update]),
                           np.array([d[1] for d in D_update]),
                           np.array([d[2] for d in D_update]),
                           np.stack([d[3] for d in D_update]),
                           np.invert(np.array([d[4] for d in D_update])))
        indt = np.where(St)[0]
        if self.nn.prep_state is not None:
            Q = self.nn.Q_predict(s_prep=s)
            Qd = self.nn_target.Q_predict(s_prep=Sd[St])
        else:
            Q = self.nn.Q_predict(s=s)
            Qd = self.nn_target.Q_predict(s=Sd[St])
        Q[range(Q.shape[0]),a] = r
        Q[indt,a[indt]] += self.gamma*np.max(Qd,1)
        self.nn.update(s,Q)

        self.step_count += 1
        if self.step_count >= self.target_steps:
            self.nn_target.assign_params(self.nn.get_params())
            self.step_count = 0

    def update(self,state,reward,done,net_update=True):
        """
        Update agent's memory and optionally update network weights

        ...
        
        Parameters
        ----------
        state : array-like
            State observation following agent selected action
        reward : float
            Reward observation following agent selected action
        done : bool
            Indicates if the new state is terminal
        net_update : bool, optional
            If the network weights should be updated, default True
        """
        if done:
            self.ep_count += 1
            self.epsilon = np.max([self.epsilon-self.d_eps,self.eps_f])

        if self.nn.prep_state is not None:
            s_prep = self.nn.sess.run(self.nn.prep_state,
                                           feed_dict={self.nn.s_input:np.concatenate([self.prev_s,state])})
            self.memory.add([s_prep[0],self.prev_a,reward,s_prep[1],done])
        else:
            self.memory.add([self.prev_s.reshape(-1),self.prev_a,reward,state.reshape(-1),done])

        if len(self.memory)+self.memory.n_demo<self.nn.k:
            return
        elif net_update:
            self.network_update()
