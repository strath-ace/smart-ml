import time
import cProfile, pstats

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