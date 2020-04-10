#############################################
##                                         ##
## code from https://github.com/rlcode/per ##
##                                         ##
#############################################

from . import networks
import random
from random import sample as _sample
import numpy as np
import numpy.random as rand
import pdb

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
	write = 0

	def __init__(self, capacity):
		self.capacity = capacity
		self.tree = np.zeros(2 * capacity - 1)
		self.data = np.zeros(capacity, dtype=object)
		self.n_entries = 0

	# update to the root node
	def _propagate(self, idx, change):
		parent = (idx - 1) // 2

		self.tree[parent] += change

		if parent != 0:
			self._propagate(parent, change)

	# find sample on leaf node
	def _retrieve(self, idx, s):
		left = 2 * idx + 1
		right = left + 1

		if left >= len(self.tree):
			return idx

		if s <= self.tree[left]:
			return self._retrieve(left, s)
		else:
			return self._retrieve(right, s - self.tree[left])

	def total(self):
		return self.tree[0]

	# store priority and sample
	def add(self, p, data):
		idx = self.write + self.capacity - 1

		self.data[self.write] = data
		self.update(idx, p)

		# TODO: adjust write to keep demo memory
		self.write += 1
		if self.write >= self.capacity:
			self.write = 0

		if self.n_entries < self.capacity:
			self.n_entries += 1

	# update priority
	def update(self, idx, p):
		change = p - self.tree[idx]

		self.tree[idx] = p
		self._propagate(idx, change)

	# get priority and sample
	def get(self, s):
		idx = self._retrieve(0, s)
		dataIdx = idx - self.capacity + 1

		return (idx, self.tree[idx], self.data[dataIdx])


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
	e = 0.01
	a = 0.6
	beta = 0.4
	beta_increment_per_sampling = 0.001

	def __init__(self, capacity):
		self.tree = SumTree(capacity)
		self.capacity = capacity

	def _get_priority(self, error):
		return (np.abs(error) + self.e) ** self.a

	def add(self, error, samp):
		p = self._get_priority(error)
		self.tree.add(p, samp)

	def sample(self, n):
		batch = []
		idxs = []
		segment = self.tree.total() / n
		priorities = []

		self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

		for i in range(n):
			a = segment * i
			b = segment * (i + 1)

			s = random.uniform(a, b)
			(idx, p, data) = self.tree.get(s)
			priorities.append(p)
			batch.append(data)
			idxs.append(idx)

		sampling_probabilities = priorities / self.tree.total()
		is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
		is_weight /= is_weight.max()

		return batch, idxs, is_weight

	def update(self, idx, error):
		p = self._get_priority(error)
		self.tree.update(idx, p)


class PerQAgent(object):
	def __init__(self,env,net_type='QNet',gamma=0.6,eps0=0.9,epsf=0.0,n_eps=400,update_steps=50,demo_memory=[],capacity=20000,**kwargs):
		self.state_size = env.state_size
		self.action_size=env.action_size
		try:
			net_module = getattr(networks,net_type)
		except AttributeError:
			raise ValueError('Invalid network type: \'{}\''.format(net_type))

		self.nn = net_module(self.state_size, self.action_size,**kwargs)
		self.nn_target = net_module(self.state_size, self.action_size,is_target=True,**kwargs)
		self.nn_target.assign_params(self.nn.get_params())

		self.memory = Memory(capacity)
		# d = <s,a,r,s',done>
		for d in demo_memory:
			pred = self.nn.Q_predict(d[0].reshape(1,-1))[0][d[1]]
			targ = d[2] if d[4] else d[2] + gamma*np.max(self.nn_target.Q_predict(d[3].reshape(1,-1)))
			self.memory.add(abs(pred-targ),tuple(d))

		self.gamma = gamma
		self.eps = eps0
		self.eps0 = eps0
		self.epsf = epsf
		self.n_eps = n_eps

		self.prev_s = []
		self.prev_a = []
		self.ep_no = 0
		self.step_count = 0
		self.C = update_steps

	def action_select(self,state):
		if rand.random(1)<self.eps:
			action=rand.randint(self.action_size)
		else:
			q_s=self.nn.Q_predict(state)
			action=np.argmax(q_s)
		self.prev_s=state
		self.prev_a=action
		return action

	def update(self,state,reward,done):
		if done:
			self.ep_no+=1
			if self.ep_no<self.n_eps:
				self.eps=float(self.eps0)-self.ep_no*(float(self.eps0)-float(self.epsf))/float(self.n_eps)
			else:
				self.eps=self.epsf
		# Update memory
		d = (self.prev_s.reshape(-1),self.prev_a,reward,state.reshape(-1),done)
		pred = self.nn.Q_predict(d[0].reshape(1,-1))[0][d[1]]
		targ = d[2] if d[4] else d[2] + self.gamma*np.max(self.nn_target.Q_predict(d[3].reshape(1,-1)))
		self.memory.add(abs(pred-targ),tuple(d))

		if self.memory.tree.n_entries>=self.nn.k:
			D_update, idxs, is_weight = self.memory.sample(self.nn.k)
		else:
			return

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
		pred = Q.copy()[np.arange(Q.shape[0]),a]
		
		Q[np.arange(Q.shape[0]),a] = r
		Q[indt,a[indt]] += self.gamma*np.max(Qd,1)
		self.nn.update(s,Q)
		
		targ = Q.copy()[np.arange(Q.shape[0]),a]
		errors = abs(pred-targ)
		for err, idx in zip(errors,idxs):
			self.memory.update(idx, err)

		self.step_count+=1
		if self.step_count >= self.C:
			self.nn_target.assign_params(self.nn.get_params())
			self.step_count=0