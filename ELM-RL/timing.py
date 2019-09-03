import time
import cProfile, pstats
from qnet_agent import QNetAgent
from eqlm_agent import EQLMAgent
from environment import Environment
from config import *

def new_profile(agent_type, n_step = 100):
	env = Environment()
	if agent_type == 'EQLM':
		agent = EQLMAgent(agent_config(),network_config(),env)
	elif agent_type == 'QNet':
		agent = QNetAgent(agent_config(),network_config(),env)
	else:
		raise ValueError('Invalid agent type')
	ps = get_profile(agent, env, n_step)
	return agent, env, ps

def get_profile(agent, env, n_step = 100):
	pr = cProfile.Profile()
	pr.enable()
	observation = env.reset()
	for step_no in range(n_step):
		action = agent.action_select(env,  observation)
		observation, reward, done, info = env.step(action)
		agent.update_net(observation, reward, done)
		if done:
			observation = env.reset()
	pr.disable()
	sortby = 'cumulative'
	ps = pstats.Stats(pr).sort_stats(sortby)
	ps.print_stats()
	return ps

def time_agent(agent,env, n_step = 100, n_run = 10):
	observation = env.reset()
	for _ in range(agent.minib+1):
		action = agent.action_select(env,  observation)
		observation, reward, done, info = env.step(action)
		agent.update_net(observation, reward, done)
		if done:
			observation = env.reset()
			
	t_vec = []
	observation = env.reset()
	for _ in range(n_run):
		ti = time.time()
		for step_no in range(n_step):
			action = agent.action_select(env,  observation)
			observation, reward, done, info = env.step(action)
			agent.update_net(observation, reward, done)
			if done:
				observation = env.reset()
		tf = time.time()
		t_vec.append(tf-ti)
		
	return t_vec