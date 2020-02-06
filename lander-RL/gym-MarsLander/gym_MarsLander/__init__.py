from gym.envs.registration import register

register(
	id='MarsLander-v0',
	entry_point='gym_MarsLander.envs:MarsLanderEnv',
	)