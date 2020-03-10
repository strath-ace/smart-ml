import numpy as np

class Attitude_constraint(object):

    def  __init__(self, attitude_parameterization, terminate_on_violation=True, 
                  attitude_limit=(np.pi/2+np.pi/8, np.pi/2-np.pi/16, np.pi/2-np.pi/16),
                  attitude_margin=(np.pi/8, np.pi/8, np.pi/8),  
                    attitude_coeff=-10.0, attitude_penalty=-100.):
        self.attitude_parameterization = attitude_parameterization
        self.attitude_margin = attitude_margin
        self.attitude_limit = attitude_limit
        self.attitude_coeff = attitude_coeff
        self.attitude_penalty = attitude_penalty
        self.terminate_on_violation = terminate_on_violation
        print('Attitude Constraint')
        self.violation_type = np.zeros(3)
        self.cnt = 0

    def get_margin(self,state,debug=False):
        att = state['attitude_321'].copy()
        if np.any(np.abs(att) > self.attitude_limit):
            margin = -1
        else:
            margin = 1
        return margin 

    def get_reward(self,state):
        att = state['attitude_321'].copy()
        yaw = att[0]
        pitch = att[1]
        roll = att[2]
        reward = self.get_r(yaw,   self.attitude_margin[0], self.attitude_limit[0]) + \
                 self.get_r(pitch, self.attitude_margin[1], self.attitude_limit[1]) + \
                 self.get_r(roll,  self.attitude_margin[2], self.attitude_limit[2])
        #print('dEBUG: ', att, reward)
        return reward 

    def get_r(self,ac,margin,limit):
        ac = np.abs(ac)
        r = 0.0
        
        tau = margin / 2
        if ac > ( limit - margin):
            err = (limit - margin) - ac
        else:
            err = 0.0 
        #print('err: ',ac, err)
        if err < 0: 
            r = -self.attitude_coeff * err 
        return r    


    def get_term_reward(self,state):
        att =  state['attitude_321']
        vio = att > self.attitude_limit
        self.violation_type += vio
        if np.any(vio):
            if self.cnt % 100 == 0:
                print('*** ATT VIO TYPE CNT: ',self.violation_type)
            self.cnt += 1
        margin = self.get_margin(state)
        if margin < 0:
            return self.attitude_penalty 
        else:
            return 0.0


        
