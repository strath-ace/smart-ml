# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import numpy as np
from .q_agents import QAgent
from .environment import Environment
from tqdm.notebook import trange

def do_run(agent, env, N_ep):
    R_ep = []
    t = trange(N_ep, desc='bar_desc', leave=True)
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
        t.set_description('Last reward: {}'.format(Rt))
        t.refresh()
    return R_ep, agent, env