import time
import cProfile, pstats
from qnet_agent import QNetAgent
from eqlm_agent import EQLMAgent, EQLMAgent2
from environment import Environment
import tensorflow as tf
from config import *
from tqdm import tqdm as p_bar

def vary_size(agent_type, k=10, N_hid=10, s_size=10, a_size=10, n_step = 500):
	# assign variables
	agentcon = agent_config()
	netcon = network_config()
	env = Environment()
	agentcon['minib'] = k
	netcon['N_hid'] = N_hid
	env.state_size = s_size
	env.action_size = a_size
	# initialise agent
	if agent_type == 'EQLM':
		agent = EQLMAgent(agentcon,netcon,env)
	elif agent_type == 'QNet':
		agent = QNetAgent(agentcon,netcon,env)
	else:
		raise ValueError('Invalid agent type')
	# run to fill memory
	observation = np.random.rand(s_size)
	for step_no in range(k):
		_ = agent.action_select(env, observation)
		observation = np.random.rand(s_size)
		agent.update_net(observation, 1, False)
	# run profiling
	pr = cProfile.Profile()
	pr.enable()
	observation = np.random.rand(s_size)
	for step_no in range(n_step):
		_ = agent.action_select(env, observation)
		observation = np.random.rand(s_size)
		agent.update_net(observation, 1, False)
	pr.disable()
	sortby = 'cumulative'
	ps = pstats.Stats(pr).sort_stats(sortby)
	ps.print_stats()
	return ps

def new_profile(agent_type, n_step = 100, show=True):
	env = Environment()
	if agent_type == 'EQLM':
		agent = EQLMAgent(agent_config(),network_config(),env)
	elif agent_type == 'QNet':
		agent = QNetAgent(agent_config(),network_config(),env)
	elif agent_type == 'EQLM2':
		agent = EQLMAgent2(agent_config(),network_config(),env)
	else:
		raise ValueError('Invalid agent type')
	ps = get_profile(agent, env, n_step, show)
	return agent, env, ps

def get_profile(agent, env, n_step = 100, show=True):
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
	if show:
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

###### Raw Timings ######

def t_elm(size_s, N_hid, size_a, k):
	agent = elm_agent(size_s, N_hid, size_a, k, H_update=False)
	pr = cProfile.Profile()
	pr.enable()
	for _ in range(500):
		A = np.random.rand(k, size_a)
		sd = np.random.rand(1, size_s)
		S = np.random.rand(k, size_s)
		agent.action_S(sd)
		agent.update_S(S,A)
	pr.disable()
	sortby = 'cumulative'
	ps = pstats.Stats(pr).sort_stats(sortby)
	return ps

def t_elmH(size_s, N_hid, size_a, k):
	agent = elm_agent(size_s, N_hid, size_a, k, H_update=True)
	pr = cProfile.Profile()
	pr.enable()
	for _ in range(500):
		A = np.random.rand(k, size_a)
		sd = np.random.rand(1, size_s)
		H = np.random.rand(k, N_hid)
		agent.action_H(sd)
		agent.update_H(H,A)
	pr.disable()
	sortby = 'cumulative'
	ps = pstats.Stats(pr).sort_stats(sortby)
	return ps

def t_elm_basic(size_s, N_hid, size_a, k):
	agent = elm_agent_basic(size_s, N_hid, size_a, k, H_update=False)
	pr = cProfile.Profile()
	pr.enable()
	for _ in range(500):
		A = np.random.rand(k, size_a)
		sd = np.random.rand(1, size_s)
		S = np.random.rand(k, size_s)
		agent.action_S(sd)
		agent.update_S(S,A)
	pr.disable()
	sortby = 'cumulative'
	ps = pstats.Stats(pr).sort_stats(sortby)
	return ps

def t_elm_basicH(size_s, N_hid, size_a, k):
	agent = elm_agent_basic(size_s, N_hid, size_a, k, H_update=True)
	pr = cProfile.Profile()
	pr.enable()
	for _ in range(500):
		A = np.random.rand(k, size_a)
		sd = np.random.rand(1, size_s)
		H = np.random.rand(k, N_hid)
		agent.action_H(sd)
		agent.update_H(H,A)
	pr.disable()
	sortby = 'cumulative'
	ps = pstats.Stats(pr).sort_stats(sortby)
	return ps

def t_qnet(size_s, N_hid, size_a, k):
	agent = q_agent(size_s, N_hid, size_a)
	pr = cProfile.Profile()
	pr.enable()
	for _ in range(500):
		S = np.random.rand(k, size_s)
		A = np.random.rand(k, size_a)
		sd = np.random.rand(1, size_s)
		agent.action(sd)
		agent.update(S,A)
	pr.disable()
	sortby = 'cumulative'
	ps = pstats.Stats(pr).sort_stats(sortby)
	return ps

###### Agents ######

class elm_agent():
    def __init__(self, s, N, a, k, H_update=True):
        self.s = tf.placeholder(shape=[None,s],dtype=tf.float32)
        self.w_in = tf.Variable(tf.random_uniform([s,N],0,0.01))
        self.b_in = tf.Variable(tf.random_uniform([1,N],0,0))
        self.W = tf.Variable(tf.random_uniform([N,a],0,0.01))
        act_fn = tf.tanh
        self.act = act_fn(tf.add(tf.matmul(self.s,self.w_in),self.b_in), name=None)
        self.a = tf.matmul(self.act,self.W)

        self.H = tf.placeholder(shape=[None,N],dtype=tf.float32) if H_update else self.act
        self.T = tf.placeholder(shape=[None,a],dtype=tf.float32)
        H_trans = tf.transpose(self.H)
        self.A_inv = tf.Variable(tf.random.uniform([N,N],0,1), use_resource=True)

        # Initialisation
        A_t1 = tf.add(tf.scalar_mul(1.0/0.05,tf.eye(N)),
            tf.matmul(H_trans,self.H))
        A_t1_inv = tf.linalg.inv(A_t1)
        W_t1 = tf.matmul(A_t1_inv,tf.matmul(H_trans,self.T))
        self.W_init = self.W.assign(W_t1)
        self.A_init = self.A_inv.assign(A_t1_inv)

        # Updating
        K1 = tf.add(tf.matmul(self.H,tf.matmul(self.A_inv,H_trans)),tf.eye(k))
        K_t = tf.subtract(tf.eye(N),
            tf.matmul(self.A_inv,tf.matmul(H_trans,tf.matmul(tf.linalg.inv(K1),self.H))))
        W_new = tf.add(tf.matmul(K_t,self.W),
            tf.matmul(tf.matmul(K_t,self.A_inv),tf.matmul(H_trans,self.T)))
        A_new = tf.matmul(K_t,self.A_inv)
        self.W_update = self.W.assign(W_new)
        self.A_update = self.A_inv.assign(A_new)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def update_H(self, H, T):
        self.sess.run((self.W_update,self.A_update),{self.H:H,self.T:T})
        
    def action_H(self, S):
        self.sess.run((self.a,self.act),{self.s:S})
        
    def update_S(self, S, T):
        self.sess.run((self.W_update,self.A_update),{self.s:S,self.T:T})
        
    def action_S(self, S):
        self.sess.run(self.a,{self.s:S})
		
class q_agent():
    def __init__(self, s, N, a):
        self.s = tf.placeholder(shape=[None,s],dtype=tf.float32)
        self.w_in = tf.Variable(tf.random_uniform([s,N],0,0.01))
        self.b_in = tf.Variable(tf.random_uniform([1,N],0,0))
        self.W = tf.Variable(tf.random_uniform([N,a],0,0.01))
        act_fn = tf.tanh
        self.act = act_fn(tf.add(tf.matmul(self.s,self.w_in),self.b_in), name=None)
        self.a = tf.matmul(self.act,self.W)

        self.ad = tf.placeholder(shape=[None,a],dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.ad - self.a))
        trainer = tf.compat.v1.train.GradientDescentOptimizer(0.01)

        grads = trainer.compute_gradients(loss,[self.W,self.w_in,self.b_in])
        cap_grads = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in grads]
        self.updateModel = trainer.apply_gradients(cap_grads)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def update(self, S, A):
        self.sess.run(self.updateModel,{self.s:S,self.ad:A})
        
    def action(self, S):
        self.sess.run(self.a,{self.s:S})
		
class elm_agent_basic():
    def __init__(self, s, N, a, k, H_update=True):
        self.s = tf.placeholder(shape=[None,s],dtype=tf.float32)
        self.w_in = tf.Variable(tf.random_uniform([s,N],0,0.01))
        self.b_in = tf.Variable(tf.random_uniform([1,N],0,0))
        self.W = tf.Variable(tf.random_uniform([N,a],0,0.01))
        act_fn = tf.tanh
        self.act = act_fn(tf.add(tf.matmul(self.s,self.w_in),self.b_in), name=None)
        self.a = tf.matmul(self.act,self.W)

        self.H = tf.placeholder(shape=[None,N],dtype=tf.float32) if H_update else self.act
        self.T = tf.placeholder(shape=[None,a],dtype=tf.float32)
        H_trans = tf.transpose(self.H)

        # Updating
#         H_inv = tf.matmul(H_trans,tf.linalg.inv(tf.matmul(H_trans,self.H)))
        H_inv = tf.matmul(H_trans,tf.linalg.inv(tf.matmul(self.H,H_trans)))
        W_new = tf.matmul(H_inv,self.T)
        self.W_update = self.W.assign(W_new)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def update_H(self, H, T):
        self.sess.run(self.W_update,{self.H:H,self.T:T})
        
    def action_H(self, S):
        self.sess.run((self.a,self.act),{self.s:S})
        
    def update_S(self, S, T):
        self.sess.run(self.W_update,{self.s:S,self.T:T})
        
    def action_S(self, S):
        self.sess.run(self.a,{self.s:S})