# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------


# from .environment import Environment
from .networks import ELMNet
# from .q_agents import QAgent

import sys
sys.path.append('../')
from QLearn.environment import Environment
from QLearn.networks import *
from QLearn.q_agents import QAgent