# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

try:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()
except ImportError:
	import tensorflow as tf
	
import numpy as np
from environment import Environment
from networks import ELMNet, QNet
from q_agents import QAgent