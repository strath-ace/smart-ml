 # This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2021 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------
"""Methods for training and testing agents"""

import numpy as np
from tqdm import trange
from .q_agents import ReplayMemory
import pickle
import pdb

def data_smooth(data,n_avg):
    """Returns data averaged over n_avg episodes for clearer plotting"""
    x_vec = np.arange(n_avg,len(data)+1,n_avg)
    data_avg = []
    for x in x_vec:
        data_avg.append(np.mean(data[x-n_avg:x]))
    return x_vec, data_avg

def train_agent(agent, env, N_ep, save_name=None, show_progress=False):
    """
    Train an agent for a fixed number of episodes

    ...

    Parameters
    ----------
    agent : QAgent
        Agent to be trained in the environment
    env : Environment
        Environment for training
    N_ep : int
        Number of episodes to train
    save_name : str or None, optional
        If str, save run data to a file named save_name after each episode, default None
    show_progress : bool, optional
        If True, shows a progress bar indicating remaining episodes, default False

    Returns
    -------
    R_ep : list
        Cumulative reward for each episode
    steps : list
        Number of steps in each episode
    agent : QAgent
        Trained agent
    env : Environment
        Environment after training
    """
    R_ep = []
    steps = []
    if show_progress:
        t = trange(N_ep, desc='bar_desc', leave=True)
    else:
        t = range(N_ep)
    for ep_no in t:
        s = env.reset()
        done = False
        Rt = 0
        n_step = 0
        while not done:
            a = agent.action_select(s)
            s, r, done, _ = env.step(a)
            agent.update(s,r,done)
            Rt += r
            n_step +=1
        R_ep.append(Rt)
        steps.append(n_step)
        if show_progress:
            if ep_no>10:
                t.set_description('R: {} Step: {}'.format(np.mean(R_ep[-10:]).round(1),n_step))
                t.refresh()
            else:
                t.set_description('R: {} Step: {}'.format(np.mean(R_ep).round(1),n_step))
                t.refresh()
        if save_name:
            data = {'params':agent.nn.get_params(),'R':R_ep,'step':steps}
            pickle.dump(data, open(save_name,'wb'))
    return R_ep, steps, agent, env

def agent_demo(agent, env, N_ep, show=False, show_progress=False):
    """
    Demonstrate a trained agent's policy over a number of episodes

    ...

    Parameters
    ----------
    agent : QAgent
        Trained agent 
    env : Environment
        Environment for demonstrating the agent
    N_ep : int
        Number of episodes to demonstrate the policy
    show : bool, optional
        If True, render the environment at each timestep, default False
    show_progress : bool, optional
        If True, shows a progress bar indicating remaining episodes, default False

    Returns
    -------
    R_ep : list
        Cumulative reward for each episode
    steps : list
        Number of steps in each episode
    """
    R_ep = []
    steps = []
    if show_progress:
        t = trange(N_ep)
    else:
        t = range(N_ep)
    for ep_no in t:
        s = env.reset()
        done = False
        Rt = 0
        n_step = 0
        while not done:
            a = agent.action_test(s)
            s, r, done, _ = env.step(a)
            if show:
                env.render()
            Rt += r
            n_step +=1
        R_ep.append(Rt)
        steps.append(n_step)
    return R_ep, steps

def heuristic_demo(H, env, N_ep, show=False):
    """
    Demonstrate a heuristic policy H over a number of episodes
    """
    R_ep = []
    steps = []
    for ep_no in trange(N_ep):
        s = env.reset()
        done = False
        Rt = 0
        n_step = 0
        while not done:
            a = H(s[0])
            s, r, done, _ = env.step(a)
            if show:
                env.render()
            Rt += r
            n_step +=1
        R_ep.append(Rt)
        steps.append(n_step)
    return R_ep, steps

def heuristic_memory_demo(H, env, N_ep, fname=None):
    """
    Create a demonstration replay buffer using a heuristic policy H
    """
    mem = ReplayMemory()
    for ep_no in trange(N_ep):
        s = env.reset()
        done = False
        while not done:
            prev_s = s
            a = H(s[0])
            s, r, done, _ = env.step(a)
            mem.add([prev_s.reshape(-1),a,r,s.reshape(-1),done])
    if fname is not None:
        pickle.dump(mem,open(fname,'wb'))
    return mem
