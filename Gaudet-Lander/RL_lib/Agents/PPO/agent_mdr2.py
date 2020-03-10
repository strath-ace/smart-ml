from utils import Logger
import scipy.signal
import signal
import numpy as np

"""

    Adapted from code written by Patrick Coady (pat-coady.github.io)

"""

class Agent(object):
    def __init__(self,policy,val_func,env,input_normalizer,logger,policy_episodes=20,policy_steps=10,gamma1=0.0,gamma2=9.995,lam=0.98,
                    normalize_advantages=True,use_tdlam=False,use_timestep=False,monitor=None, animate=False):
        self.env = env
        self.monitor = monitor
        self.policy_steps = policy_steps
        self.logger = logger
        self.use_tdlam = use_tdlam
        self.use_timestep = use_timestep
        self.policy = policy 
        self.val_func = val_func
        self.input_normalizer = input_normalizer
        self.policy_episodes = policy_episodes
        self.animate = animate 
        self.normalize_advantages = normalize_advantages 
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        self.lam = lam
        self.global_steps = 0
        
        """ 

            Args:
                policy:                 policy object with update() and sample() methods
                val_func:               value function object with fit() and predict() methods
                env:                    environment
                input_normalizer:       scaler object with apply(), reverse(), and update() methods
                logger:                 Logger object

                policy_episodes:        number of episodes collected before update
                policy_steps:           minimum number of steps before update
                    (will update when either episodes > policy_episodes or steps > policy_steps)

                gamma:                  discount rate
                lam:                    lambda for GAE calculation
                normalize_advantages:   boolean, normalizes advantages if True
                use_tdlam:              boolean, True uses TD lambda target for value function, else Monte Carlo 
                use_timestep:           boolean, True enables time step feature which sometimes works better than a 
                                        low discount rate for continuing tasks with per-step rewards (like Mujoco envs)
                monitor:                A monitor object like RL_stats to plot interesting stats as learning progresses
                                        Monitor object implements update_episode() and show() methods 
                animate:                boolean, True uses env.render() method to animate episode

        """ 
  
    def run_episode(self):
        """

        Returns: 4-tuple of NumPy arrays
            observes: shape = (episode len, obs_dim)
            actions: shape = (episode len, act_dim)
            rewards: shape = (episode len,)
            unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
        """
        obs = self.env.reset()
        observes, actions, rewards1, rewards2, unscaled_obs  =  [], [], [], [], []
        done = False
        step = 0.0
        while not done:
            if self.animate:
                self.env.render()
            obs = obs.astype(np.float64).reshape((1, -1))
            unscaled_obs.append(obs.copy())
            if self.input_normalizer is not None:
                obs = self.input_normalizer.apply(obs)
            if self.use_timestep:
                obs = np.append(obs, [[step]], axis=1)  # add time step feature
            observes.append(obs)
            action, env_action = self.policy.sample(obs)# .reshape((1, -1)).astype(np.float64) #[:,0:-1])
            actions.append(action)
            obs, reward, done, reward_info = self.env.step(env_action)
            reward1 = reward[0]
            reward2 = reward[1]
            if not isinstance(reward1, float):
                reward1 = np.asscalar(reward1)
            if not isinstance(reward1, float):
                reward2 = np.asscalar(reward2)
            rewards1.append(reward1)
            rewards2.append(reward2)
            step += 1e-3  # increment time step feature
        #logger.log({'Score': sum_rewards})
        return (np.concatenate(observes), np.concatenate(actions), np.array(rewards1, dtype=np.float64), 
                np.array(rewards2, dtype=np.float64), np.concatenate(unscaled_obs))


    def run_policy(self,episode_cnt,warmup=False):
        """ Run policy and collect data for a minimum of min_steps and min_episodes
        Args:
            episode_cnt: current episode number, used for logging stats     

        Returns: list of trajectory dictionaries, list length = number of episodes
            'observes' : NumPy array of states from episode
            'actions' : NumPy array of actions from episode
            'rewards' : NumPy array of (un-discounted) rewards from episode
            'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
        """
        total_steps = 0
        e_cnt = 0
        trajectories = []
        #for e in range(self.policy_episodes):
        while e_cnt <= self.policy_episodes or total_steps < self.policy_steps:
            observes, actions, rewards1, rewards2, unscaled_obs  = self.run_episode()
            if self.monitor is not None and not warmup:
                self.monitor.update_episode(np.sum(rewards1) + np.sum(rewards2),  observes.shape[0])
            total_steps += observes.shape[0]
            trajectory = {'observes': observes,
                          'actions': actions,
                          'rewards1': rewards1,
                          'rewards2': rewards2,
                          'unscaled_obs': unscaled_obs}
            trajectories.append(trajectory)
            e_cnt += 1
        unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
        if self.input_normalizer is not None:
            self.input_normalizer.update(unscaled)  # update running statistics for scaling observations

        self.add_value(trajectories)  # add estimated values to episodes
        self.add_disc_sum_rew(trajectories, self.gamma1, self.gamma2)  # calculated discounted sum of Rs
        self.add_gae(trajectories, self.gamma1, self.gamma2, self.lam)  # calculate advantage
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = self.build_train_set(trajectories)

        if not warmup:
            self.policy.update(observes, actions, advantages, self.logger)  # update policy
            self.val_func.fit(observes, disc_sum_rew, self.logger)  # update value function
            self.log_batch_stats(observes, actions, advantages, disc_sum_rew, episode_cnt)
            self.global_steps += total_steps
            self.logger.log({'_MeanReward': np.mean([t['rewards1'].sum() + t['rewards2'].sum() for t in trajectories]),
                         '_StdReward': np.std([t['rewards1'].sum() + t['rewards2'].sum() for t in trajectories]),
                         '_MinReward': np.min([t['rewards1'].sum() + t['rewards2'].sum()  for t in trajectories]),
                         'Steps': total_steps,
                         'TotalSteps' : self.global_steps})
            if self.monitor is not None: 
                self.monitor.show()
        return trajectories

    def train(self,train_episodes, train_samples=None):
        _ = self.run_policy(-1,warmup=True)
        print('*** SCALER WARMUP COMPLETE *** ')
        print(np.sqrt(self.input_normalizer.vars))
        episode = 0
       
        if train_samples is not None:
            while self.global_steps < train_samples:
                trajectories = self.run_policy(episode)
                self.logger.write(display=True)
                episode += len(trajectories)
        else: 
            while episode < train_episodes: 
                trajectories = self.run_policy(episode)
                self.logger.write(display=True)
                episode += len(trajectories)
            
  
    def discount(self,x, gamma):
        """ Calculate discounted forward sum of a sequence at each point """
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


    def add_disc_sum_rew(self,trajectories, gamma1, gamma2):
        """ Adds discounted sum of rewards to all time steps of all trajectories

        Args:
            trajectories: as returned by run_policy()
            gamma: discount

        Returns:
            None (mutates trajectories dictionary to add 'disc_sum_rew')
        """
        for trajectory in trajectories:
            if gamma1 < 0.999:  # don't scale for gamma ~= 1
                rewards1 = trajectory['rewards1'] * (1 - gamma1)
            else:
                rewards1 = trajectory['rewards1']

            if gamma2 < 0.999:  # don't scale for gamma ~= 1
                rewards2 = trajectory['rewards2'] * (1 - gamma2)
            else:
                rewards2 = trajectory['rewards2']

            disc_sum_rew1 = self.discount(rewards1, gamma1)
            disc_sum_rew2 = self.discount(rewards2, gamma2)

            trajectory['disc_sum_rew'] = disc_sum_rew1 + disc_sum_rew2


    def add_value(self,trajectories):
        """ Adds estimated value to all time steps of all trajectories

        Args:
            trajectories: as returned by run_policy()
            val_func: object with predict() method, takes observations
                and returns predicted state value

        Returns:
            None (mutates trajectories dictionary to add 'values')
        """
        for trajectory in trajectories:
            observes = trajectory['observes']
            values = self.val_func.predict(observes)
            trajectory['values'] = values


    def add_gae(self,trajectories, gamma1, gamma2, lam):
        """ Add generalized advantage estimator.
        https://arxiv.org/pdf/1506.02438.pdf

        Args:
            trajectories: as returned by run_policy(), must include 'values'
                key from add_value().
            gamma: reward discount
            lam: lambda (see paper).
                lam=0 : use TD residuals
                lam=1 : A =  Sum Discounted Rewards - V_hat(s)

        Returns:
            None (mutates trajectories dictionary to add 'advantages')
        """
        for trajectory in trajectories:
            advantages = trajectory['disc_sum_rew'] - trajectory['values']
            trajectory['advantages'] = advantages

    def build_train_set(self,trajectories):
        """

        Args:
            trajectories: trajectories after processing by add_disc_sum_rew(),
                add_value(), and add_gae()

        Returns: 4-tuple of NumPy arrays
            observes: shape = (N, obs_dim)
            actions: shape = (N, act_dim)
            advantages: shape = (N,)
            disc_sum_rew: shape = (N,)
        """
        observes = np.concatenate([t['observes'] for t in trajectories])
        actions = np.concatenate([t['actions'] for t in trajectories])
        disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
        advantages = np.concatenate([t['advantages'] for t in trajectories])
        # normalize advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        else:
            advantages = advantages - advantages.mean()

        return observes, actions, advantages, disc_sum_rew

    def log_batch_stats(self,observes, actions, advantages, disc_sum_rew, episode):
        """ Log various batch statistics """
        self.logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
            })

