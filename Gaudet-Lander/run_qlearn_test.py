import sys
import numpy as np
from tqdm import trange
import pickle
from multiprocessing import Pool, cpu_count

sys.path.append('../')

import QLearn
from env_gaudet_lander import LanderEnvironment


def demo_run(run_no):
	N_ep = 50000
	env = LanderEnvironment()
	fname = 'test_demo_25_3_{}.pkl'.format(run_no)
	dmem=pickle.load(open('demo_disc.pkl','rb'))
	agent=QLearn.QAgent(env,net_type='MLPQNet',hidden_layers=[160,80],alpha=1e-4,gamma=0.99,eps0=0.25,n_eps=40000,
					   minibatch_size=100,memory_size=500000,demo_memory=dmem,n_demo=100)
	R, _, _, _ = QLearn.do_run(agent, env, N_ep, save_name = fname, show_progress=True)
	return R

if __name__ == '__main__':
	n_run = 8
	n_process = np.min([cpu_count(),n_run])
	p = Pool(processes=n_process)
	R_ep_runs = p.map_async(demo_run, range(n_run))
	R_runs = R_ep_runs.get()
	p.close()
	p.join()
	pickle.dump(R_runs,open('all_rewards2.pkl','wb'))