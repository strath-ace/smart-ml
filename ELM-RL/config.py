import numpy as np

def network_config():
	netcon = {}
	netcon['alpha'] = 0.01
	netcon['gamma_reg'] = 0.0621
	netcon['clip_norm'] = 1.0
	netcon['update_steps'] = 15
	netcon['N_hid'] = 11
	netcon['activation'] = 'sigmoid'
	netcon['init_mag'] = 0.1
	return netcon


def agent_config():
	agentcon = {}
	agentcon['gamma'] = 0.5
	agentcon['eps0'] = 0.782
	agentcon['epsf'] = 0.0
	agentcon['n_eps'] = 410
	agentcon['minib'] = 10
	agentcon['max_mem'] = 10000
	return agentcon